import os
import sys
import json
import re
from typing import List, Dict, Any
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
# from areal.dataset import get_custom_dataset  # 注释掉原有的 dataset loader
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow


# ==========================================
# 1. 辅助函数：提取与清洗 (来自你的参考代码)
# ==========================================

def _normalize_text(text: str) -> str:
    """
    文本标准化函数：去除所有空白字符并转换为小写
    用于后续的字符串比较，提高匹配准确性
    """
    return "".join(str(text).lower().split())

def _extract_answer_from_box(text: str) -> str:
    """
    从文本中提取答案的核心函数
    支持多种格式的答案提取：
    1. LaTeX格式：\\boxed{...}
    2. 自定义标签：<|answer|>...</|answer|>
    3. HTML标签：<answer>...</answer>
    4. 智能清理：移除思考标签<think>...</think>等无关内容
    """
    # 首先移除思考标签，这些标签包含解题思路但不是最终答案
    text_wo_think = re.sub(r"<\|think\|>.*?</\|think\|>", "", text, flags=re.DOTALL)
    text_wo_think = re.sub(r"<think>.*?</think>", "", text_wo_think, flags=re.DOTALL | re.IGNORECASE)

    # 尝试提取LaTeX格式的boxed答案：\\boxed{...}
    # 使用正则表达式匹配嵌套的大括号，如 \\boxed{A} 或 \\boxed{A \text{ and } B}
    boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    m = re.search(boxed_pattern, text_wo_think)
    if m:
        return m.group(1).strip()  # 返回提取到的内容并去除首尾空格
    
    # 尝试提取自定义答案标签：<|answer|>...</|answer|>
    m = re.search(r"<\|answer\|>(.*?)</\|answer\|>", text_wo_think, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    # 尝试提取HTML风格答案标签：<answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", text_wo_think, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 如果以上格式都找不到，作为保底方案，清理所有标签后返回纯文本
    no_pipe_tags = re.sub(r"<\|/?[^>]*\|>", "", text_wo_think)  # 移除 <|...|> 格式标签
    no_all_tags = re.sub(r"</?[^>]+>", "", no_pipe_tags)        # 移除 <...> 格式标签
    return no_all_tags.strip()

# ==========================================
# 2. 新增 Reward Function: ABCD 匹配
# ==========================================

def abcd_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    """
    ABCD选择题奖励函数：严格匹配预测答案与标准答案
    主要用于选择题场景，对答案格式要求较高
    
    参数：
    - prompt: 输入的提示文本
    - completions: 模型生成的完成文本（通常为列表）
    - prompt_ids: 输入token的ID
    - completion_ids: 生成token的ID  
    - answer: 标准答案（Ground Truth）
    - **kwargs: 其他可选参数
    
    返回值：匹配成功返回1.0，失败返回0.0
    """
    # 从模型生成的文本中提取答案（通常取第一个生成结果）
    completion_text = completions[0] if isinstance(completions, list) else completions
    pred_answer = _extract_answer_from_box(completion_text)  # 提取boxed中的答案
    
    # 对预测答案和标准答案进行标准化处理（去空格、转小写）
    pred_norm = _normalize_text(pred_answer)
    gt_norm = _normalize_text(answer)

    # 如果预测答案为空，直接给0分
    if not pred_norm:
        return 0.0
    
    # 严格匹配：只有当预测答案与标准答案完全相同时才给满分
    if pred_norm == gt_norm:
        return 1.0
    
    # 其他情况给0分
    return 0.0

def format_and_accuracy_reward_fn(prompt, completions, answer, **kwargs):
    """
    综合奖励函数：格式奖励 + 准确性奖励
    鼓励模型既要有正确的格式（包含boxed），又要有正确的答案
    
    参数：
    - prompt: 输入提示文本
    - completions: 模型生成的文本
    - answer: 标准答案
    - **kwargs: 其他参数
    
    返回值：格式奖励(0.4) + 准确性奖励(1.0) 的综合得分
    """
    # 定义奖励权重
    R_FORMAT = 0.4  # 格式正确奖励：0.4分
    R_CORRECT = 1.0 # 答案正确奖励：1.0分
    
    # 获取模型生成的完整文本
    completion_text = completions[0] if isinstance(completions, list) else completions
    
    # 初始化当前奖励为0
    current_reward = 0.0
    
    # --- 1. 格式奖励部分 ---
    # 检查生成文本是否包含LaTeX的boxed格式，有格式奖励0.4分
    if "\\boxed{" in completion_text:
        current_reward += R_FORMAT
        
    # --- 2. 准确性奖励部分 ---
    # 从生成文本中提取预测答案
    pred_answer = _extract_answer_from_box(completion_text)
    
    # 标准化预测答案和标准答案，便于比较
    pred_norm = _normalize_text(pred_answer)
    gt_norm = _normalize_text(answer)
    
    # 只有当两个答案都不为空时才进行比较
    if pred_norm and gt_norm:
        # 完全匹配：预测答案与标准答案完全相同
        if pred_norm == gt_norm:
            current_reward += R_CORRECT
        # 针对选择题的额外容错匹配：当标准答案是单个字母时
        elif len(gt_norm) == 1 and gt_norm in pred_norm:
            # 检查是否是标准格式，如 "D"、"optionD"、"choiceD" 等
            if pred_norm in [gt_norm, f"option{gt_norm}", f"choice{gt_norm}"]:
                current_reward += R_CORRECT

    # 返回综合奖励得分
    return current_reward

# ==========================================
# 3. 新增 Dataset Class: 处理 JSONL 数据文件
# ==========================================
class JsonlDataset(Dataset):
    """
    自定义数据集类：用于处理JSONL格式的多模态数据
    支持从JSONL文件加载数据，构建提示文本，提取标准答案
    """
    def __init__(self, data_path, tokenizer, max_length=2048):
        """
        初始化数据集
        
        参数：
        - data_path: JSONL文件路径
        - tokenizer: 分词器对象
        - max_length: 最大序列长度限制
        """
        self.tokenizer = tokenizer  # 用于文本分词
        self.data = []              # 存储加载的数据
        self.max_length = max_length # 最大序列长度
        self.base_dir = os.path.dirname(data_path) # 数据文件所在目录，用于处理相对路径图片
        
        # 从JSONL文件加载数据
        print(f"正在从 {data_path} 加载数据...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    try:
                        # 解析每一行为JSON对象并添加到数据列表
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 如果解析失败，跳过该行
                        continue
        print(f"成功加载 {len(self.data)} 个样本.")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def _get_image_path(self, sample):
        """
        获取图片路径的辅助函数
        处理相对路径和绝对路径的转换
        """
        # 从样本中获取图片路径信息
        image_path = sample.get("image", None)
        if image_path:
            # 如果是多张图片（列表形式），取第一张
            if isinstance(image_path, list):
                image_path = image_path[0]
            
            # 如果路径不是绝对路径，尝试拼接数据集所在目录
            if not os.path.isabs(image_path):
                # 构造可能的完整路径
                potential_path = os.path.join(self.base_dir, image_path)
                if os.path.exists(potential_path):
                    # 如果路径存在，更新为绝对路径
                    image_path = potential_path
                else:
                    # 如果拼接后不存在，保持原路径（可能是相对于当前工作目录）
                    pass
        return image_path

    def _build_prompt_text(self, sample):
        """
        构建模型输入提示文本的函数
        支持多种数据格式：LLaVA格式和OpenAI格式
        """
        # 定义提示后缀：要求模型以\\boxed{}格式输出答案
        prompt_suffix = "\n请逐步回答问题，最后必须以如下格式给出答案：\\boxed{your_answer}。"
        
        user_content = ""  # 存储用户输入内容
        
        # 1. 尝试解析LLaVA格式的对话数据
        if "conversations" in sample:
            for turn in sample["conversations"]:
                if turn.get("from") == "human":  # 找到人类用户的消息
                    val = turn.get("value", "")  # 获取消息内容
                    # 移除<image>标签，因为图片路径会单独传递给推理引擎
                    val = val.replace("<image>", "").strip()
                    user_content = val
                    break  # 找到用户消息后退出循环
                    
        # 2. 尝试解析OpenAI格式的消息数据
        elif "messages" in sample:
            for message in sample["messages"]:
                if message.get("role") == "user":  # 找到用户角色的消息
                    val = message.get("content", "")  # 获取内容
                    if isinstance(val, str):
                        # 如果内容是字符串，移除图片标签
                        val = val.replace("<image>", "").strip()
                        user_content = val
                    elif isinstance(val, list):
                        # 如果内容是列表（多模态内容），逐个处理
                        for part in val:
                            if part.get("type") == "text":  # 只处理文本部分
                                user_content += part.get("text", "")
                    break  # 找到用户消息后退出循环
        
        # 在用户内容后添加提示后缀
        full_user_content = user_content + prompt_suffix

        # 使用分词器的聊天模板构建完整提示文本
        if hasattr(self.tokenizer, "apply_chat_template"):
            # 构建消息列表
            messages = [{"role": "user", "content": full_user_content}]
            try:
                # 应用聊天模板，不进行分词，添加生成提示
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # 如果聊天模板应用失败，使用备用格式
                prompt_text = f"User: {full_user_content}\nAssistant:"
        else:
            # 如果分词器不支持聊天模板，使用简单格式
            prompt_text = f"User: {full_user_content}\nAssistant:"
            
        return prompt_text

    def _extract_ground_truth(self, sample):
        """
        从样本中提取标准答案的函数
        优先使用gt_answer字段，否则从对话历史中提取
        """
        # 1. 优先使用数据集中明确标注的标准答案字段
        if "gt_answer" in sample:
            return str(sample["gt_answer"])
            
        # 2. 如果没有gt_answer字段，从对话历史中查找助手的回答
        if "conversations" in sample:
            for turn in sample["conversations"]:
                if turn.get("from") in ("gpt", "assistant"):  # 查找助手的回答
                    return str(turn.get("value", ""))
        elif "messages" in sample:
            for message in sample["messages"]:
                if message.get("role") == "assistant":  # 查找助手角色的消息
                    return str(message.get("content", ""))
        return ""

    def __getitem__(self, index):
        """
        获取数据集中的单个样本
        返回包含输入ID、标准答案和图片路径的字典
        """
        # 获取指定索引的样本
        sample = self.data[index]
        
        # 获取图片路径
        image_path = self._get_image_path(sample)
        # 构建提示文本
        prompt_text = self._build_prompt_text(sample)
        
        # 使用分词器将提示文本转换为token ID
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        # 如果序列长度超过最大限制，截取最后max_length个token
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]

        # 提取标准答案
        raw_answer = self._extract_ground_truth(sample)
        # 尝试从原始答案中提取boxed格式的答案
        clean_answer = _extract_answer_from_box(raw_answer)
        if not clean_answer: 
            # 如果提取失败，使用原始答案并去除首尾空格
            clean_answer = raw_answer.strip()

        # 返回包含输入、答案和图片路径的字典
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),  # 转换为PyTorch张量
            "answer": clean_answer,      # 标准答案
            "image_path": image_path,    # 图片路径
        }

# ==========================================
# 4. 主训练循环函数
# ==========================================

def main(args):
    """
    主训练函数：执行完整的PPO训练流程
    包括数据加载、模型初始化、训练循环等
    """
    # 加载配置文件
    config, _ = load_expr_config(args, GRPOConfig)

    # 获取当前进程的排名
    rank = int(os.getenv("RANK", "0"))
    # 加载分词器
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # 设置随机种子，确保训练结果可复现
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    # 从配置中获取分配模式
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    # 获取训练并行策略
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None  # 确保并行策略存在

    # 初始化训练引擎（PPO Actor）
    actor = FSDPPPOActor(config=config.actor)
    # 创建并行处理组
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # ------------------------------------------------------------------
    # 修改点：使用自定义的JsonlDataset加载数据
    # ------------------------------------------------------------------
    # 从配置文件中获取训练数据路径，加载JSONL格式的训练数据集
    train_dataset = JsonlDataset(
        data_path=config.train_dataset.data_path,  # 训练数据路径
        tokenizer=tokenizer,                       # 分词器
        max_length=config.train_dataset.get("max_length", 2048)  # 最大长度，默认2048
    )
    
    # 同样方式加载验证数据集
    valid_dataset = JsonlDataset(
        data_path=config.valid_dataset.data_path,  # 验证数据路径
        tokenizer=tokenizer,
        max_length=config.valid_dataset.get("max_length", 2048)  # 最大长度
    )
    # ------------------------------------------------------------------

    # 创建训练数据加载器
    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,              # 当前进程排名
        world_size=actor.data_parallel_world_size, # 并行总进程数
        dataset_config=config.train_dataset,       # 数据集配置
    )
    # 创建验证数据加载器
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
    )
    
    # 创建微调规格对象，包含训练轮数、数据集大小等信息
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,  # 总训练轮数
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,  # 数据集总大小
        train_batch_size=config.train_dataset.batch_size,  # 训练批次大小
    )

    # 初始化推理引擎（用于生成样本）
    rollout = RemoteSGLangEngine(config.rollout)
    # 初始化训练时的推理引擎
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    # 初始化验证时的推理引擎（副本）
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    
    # 注意：验证时不控制offpolicyness（策略偏离度）
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    # 从FSDP跨设备通信中获取权重更新元信息
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    # 初始化actor模型
    actor.initialize(None, ft_spec)
    # 将actor连接到推理引擎
    actor.connect_engine(rollout, weight_update_meta)

    # 如果需要KL散度控制（防止策略偏离过大），初始化参考模型
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # ------------------------------------------------------------------
    # 修改点：使用自定义的奖励函数替换默认的workflow
    # ------------------------------------------------------------------
    # 创建训练时的工作流，使用自定义奖励函数
    workflow = RLVRWorkflow(
        reward_fn=format_and_accuracy_reward_fn,  # 使用格式+准确性奖励函数
        gconfig=config.gconfig,                   # 生成配置
        tokenizer=tokenizer,                      # 分词器
        enable_thinking=False,                    # 是否启用思考标签处理
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"  # 生成结果保存目录
        ),
    )
    
    # 创建验证时的工作流，同样使用自定义奖励函数
    eval_workflow = RLVRWorkflow(
        reward_fn=format_and_accuracy_reward_fn,  # 使用相同的奖励函数
        gconfig=config.gconfig.new(temperature=0.6),  # 验证时使用稍低的温度
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",        # 验证统计范围
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"  # 验证生成结果保存目录
        ),
    )
    # ------------------------------------------------------------------

    # 初始化各种训练辅助组件
    saver = Saver(config.saver, ft_spec)           # 模型保存器
    stats_logger = StatsLogger(config, ft_spec)    # 统计日志记录器
    evaluator = Evaluator(config.evaluator, ft_spec)  # 评估器

    # 恢复处理器：用于从断点恢复训练
    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor, saver, evaluator, stats_logger, train_dataloader,
        inference_engine=rollout, weight_update_meta=weight_update_meta,
    )
    # 获取恢复训练的起始步数
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    # 计算训练总步数
    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    # 开始主训练循环
    for global_step in range(start_step, max_steps):
        # 计算当前轮数和步数
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        # 创建步信息对象
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        # --- 1. Rollout阶段：生成样本 ---
        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,  # 分组大小
                workflow=workflow,                     # 使用自定义工作流
                should_accept_fn=lambda sample: True, # 接受所有样本
            )

        # --- 2. 重新计算对数概率 ---
        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)      # 计算对数概率
                batch["prox_logp"] = logp            # 存储到批次中
                log_gpu_stats("recompute logp")      # 记录GPU状态

        # --- 3. 计算参考模型的对数概率 ---
        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)  # 参考模型概率
                log_gpu_stats("ref logp")

        # --- 4. 计算优势函数 ---
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)          # 计算优势值
            log_gpu_stats("compute advantages")

        # --- 5. PPO更新步骤 ---
        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)                  # 执行PPO更新
            actor.step_lr_scheduler()               # 更新学习率
            log_gpu_stats("ppo update")

        # 暂停rollout引擎
        rollout.pause()

        # --- 6. 更新权重 ---
        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)  # 更新模型权重

            # 同步所有相关引擎的版本号
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        # --- 7. 保存模型 ---
        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        # --- 8. 保存恢复点 ---
        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor, step_info, saver, evaluator, stats_logger, train_dataloader,
                tokenizer=tokenizer,
            )

        # 同步所有进程
        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # --- 9. 验证阶段 ---
        with stats_tracker.record_timing("eval"):
            def evaluate_fn():
                """
                验证函数：在验证集上评估模型性能
                """
                if actor.is_data_parallel_head():  # 只在主进程中执行
                    cnt = 0
                    for data in valid_dataloader:
                        # 处理批次数据：JsonlDataset返回字典格式
                        input_ids_batch = data["input_ids"]    # 输入ID批次
                        answers_batch = data["answer"]         # 答案批次
                        images_batch = data["image_path"]      # 图片路径批次
                        
                        batch_size = len(input_ids_batch)
                        # 逐个处理批次中的样本
                        for i in range(batch_size):
                            item = {
                                "input_ids": input_ids_batch[i],    # 单个输入ID
                                "answer": answers_batch[i],         # 单个答案
                                "image_path": images_batch[i]       # 单个图片路径
                            }
                            # 提交样本到验证推理引擎
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    # 等待所有验证任务完成
                    eval_rollout.wait(cnt, timeout=None)
                # 同步所有进程
                current_platform.synchronize()
                dist.barrier(group=actor.cpu_group)

            # 执行验证
            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        # 验证后同步
        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # --- 10. 记录统计信息 ---
        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # 恢复rollout引擎
        rollout.resume()

    # 训练结束后清理资源
    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    # 程序入口：解析命令行参数并启动主函数
    main(sys.argv[1:])