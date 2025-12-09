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
    """去除空白字符并转小写"""
    return "".join(str(text).lower().split())

def _extract_answer_from_box(text: str) -> str:
    """提取 \\boxed{} 中的内容"""
    # 移除 think 标签
    text_wo_think = re.sub(r"<\|think\|>.*?</\|think\|>", "", text, flags=re.DOTALL)
    text_wo_think = re.sub(r"<think>.*?</think>", "", text_wo_think, flags=re.DOTALL | re.IGNORECASE)

    # 提取 boxed
    boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    m = re.search(boxed_pattern, text_wo_think)
    if m:
        return m.group(1).strip()
    
    # 备选 tag
    m = re.search(r"<\|answer\|>(.*?)</\|answer\|>", text_wo_think, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    m = re.search(r"<answer>(.*?)</answer>", text_wo_think, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 清理所有标签作为保底
    no_pipe_tags = re.sub(r"<\|/?[^>]*\|>", "", text_wo_think)
    no_all_tags = re.sub(r"</?[^>]+>", "", no_pipe_tags)
    return no_all_tags.strip()

# ==========================================
# 2. 新增 Reward Function: ABCD 匹配
# ==========================================

def abcd_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    """
    Reward 函数：判断模型输出的 boxed 内容是否与 answer (Ground Truth) 一致。
    针对多选题 (A/B/C/D)。
    """
    # 1. 提取模型生成的答案
    # completions 通常是一个列表，这里取第一个（RLVR通常每次只看一个生成结果进行打分）
    completion_text = completions[0] if isinstance(completions, list) else completions
    pred_answer = _extract_answer_from_box(completion_text)
    
    # 2. 规范化
    pred_norm = _normalize_text(pred_answer)
    gt_norm = _normalize_text(answer)

    # 3. 比较
    # 如果预测为空，直接0分
    if not pred_norm:
        return 0.0
    
    # 只要 Ground Truth 包含在预测中（例如 GT="A", Pred="A" 或 Pred="OptionA"），或者完全相等
    # 对于多选题，通常要求严格匹配或者单个字母匹配
    if pred_norm == gt_norm:
        return 1.0
    
    return 0.0

def format_and_accuracy_reward_fn(prompt, completions, answer, **kwargs):
    """
    Reward = Format Reward + Accuracy Reward
    """
    # 权重配置
    R_FORMAT = 0.4  # 格式正确给 0.4 分
    R_CORRECT = 1.0 # 答案正确给 1.0 分
    
    completion_text = completions[0] if isinstance(completions, list) else completions
    
    current_reward = 0.0
    
    # --- 1. Format Reward ---
    # 检查是否包含 boxed 命令
    if "\\boxed{" in completion_text:
        current_reward += R_FORMAT
        
    # --- 2. Accuracy Reward ---
    # 提取模型预测内容
    pred_answer = _extract_answer_from_box(completion_text)
    
    # 规范化文本 (去空、转小写)
    pred_norm = _normalize_text(pred_answer)
    gt_norm = _normalize_text(answer)
    
    # 比较
    if pred_norm and gt_norm:
        # 如果完全匹配
        if pred_norm == gt_norm:
            current_reward += R_CORRECT
        # 针对选择题的额外鲁棒性检查 (例如 GT="D", Pred="OptionD")
        elif len(gt_norm) == 1 and gt_norm in pred_norm:
             # 这种比较比较宽松，如果 pred_norm 是 "answer is d"，也会匹配
             # 如果为了严谨，可以只检查 pred_norm == gt_norm
             if pred_norm in [gt_norm, f"option{gt_norm}", f"choice{gt_norm}"]:
                 current_reward += R_CORRECT

    return current_reward

# ==========================================
# 3. 新增 Dataset Class: 处理 JSONL 和 Prompt
# ==========================================
class LazyJsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_dir = os.path.dirname(data_path)
        self.offsets = [0]
        
        print(f"Indexing dataset: {data_path} ...")
        # 第一次扫描：只记录每一行的起始位置，不加载内容
        with open(data_path, "rb") as f:
            while f.readline():
                self.offsets.append(f.tell())
        # 移除最后多余的一个 offset (文件末尾)
        self.offsets.pop()
        
        print(f"Indexed {len(self.offsets)} samples. (Lazy Loading Mode)")

    def __len__(self):
        return len(self.offsets)

    def _get_image_path(self, sample):
        """获取图片绝对路径"""
        image_path = sample.get("image", None)
        if image_path:
            # 如果是列表（多图），取第一张或根据需求处理
            if isinstance(image_path, list):
                image_path = image_path[0]
            
            # 如果路径不是绝对路径，尝试拼接 dataset 所在目录
            if not os.path.isabs(image_path):
                 # 这里假设图片是相对于 jsonl 文件或者某个 root 目录
                 # 你可以根据实际情况修改这里的拼接逻辑
                 potential_path = os.path.join(self.base_dir, image_path)
                 if os.path.exists(potential_path):
                     image_path = potential_path
                 else:
                     # 如果拼接后不存在，保持原样（可能是相对于 cwd）
                     pass
        return image_path

    def _build_prompt_text(self, sample):
        """构建 Prompt"""
        # 你的 Prompt 后缀要求
        prompt_suffix = "\nPlease answer the question step by step, you must give the answer in this form finally: \\boxed{your_answer}."
        
        user_content = ""
        
        # 1. 尝试解析 conversations (LLaVA 格式)
        if "conversations" in sample:
            for turn in sample["conversations"]:
                if turn.get("from") == "human":
                    val = turn.get("value", "")
                    # 移除 <image> 标签，因为 image_path 会单独传给 engine
                    val = val.replace("<image>", "").strip()
                    user_content = val
                    break
                    
        # 2. 尝试解析 messages (OpenAI 格式)
        elif "messages" in sample:
            for message in sample["messages"]:
                if message.get("role") == "user":
                    val = message.get("content", "")
                    if isinstance(val, str):
                        val = val.replace("<image>", "").strip()
                        user_content = val
                    elif isinstance(val, list):
                        # 处理 content 为 list 的情况 (multimodal content)
                        for part in val:
                            if part.get("type") == "text":
                                user_content += part.get("text", "")
                    break
        
        full_user_content = user_content + prompt_suffix

        # 构建 Chat Template
        # 如果 tokenizer 支持 chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": full_user_content}]
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback format
                prompt_text = f"User: {full_user_content}\nAssistant:"
        else:
            prompt_text = f"User: {full_user_content}\nAssistant:"
            
        return prompt_text

    def _extract_ground_truth(self, sample):
        """优先从 gt_answer 字段提取，否则解析历史"""
        # 1. 优先使用 dataset 中明确标注的 gt_answer
        if "gt_answer" in sample:
            return str(sample["gt_answer"])
            
        # 2. Fallback: 从 conversation 历史中找 assistant 的回答
        if "conversations" in sample:
            for turn in sample["conversations"]:
                if turn.get("from") in ("gpt", "assistant"):
                    return str(turn.get("value", ""))
        elif "messages" in sample:
            for message in sample["messages"]:
                if message.get("role") == "assistant":
                    return str(message.get("content", ""))
        return ""

    def __getitem__(self, index):
        offset = self.offsets[index]
        
        # 在获取数据时才读取文件
        with open(self.data_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                # 容错处理，返回空数据或随机数据防止崩溃
                # 实际生产中建议 print error
                return self.__getitem__((index + 1) % len(self))

        image_path = self._get_image_path(sample)
        prompt_text = self._build_prompt_text(sample)
        
        input_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]

        raw_answer = self._extract_ground_truth(sample)
        clean_answer = _extract_answer_from_box(raw_answer)
        if not clean_answer: 
            clean_answer = raw_answer.strip()

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "answer": clean_answer, 
            "image_path": image_path,
        }
# ==========================================
# 4. Main Loop 修改
# ==========================================

def main(args):
    config, _ = load_expr_config(args, GRPOConfig)

    rank = int(os.getenv("RANK", "0"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # ------------------------------------------------------------------
    # 修改点：使用自定义的 JsonlDataset
    # ------------------------------------------------------------------
    # 假设 config.train_dataset.data_path 是 jsonl 文件的路径
    train_dataset = LazyJsonlDataset(
        data_path=config.train_dataset.data_path, 
        tokenizer=tokenizer,
        max_length=config.train_dataset.get("max_length", 2048)
    )
    
    # 假设 config.valid_dataset.data_path 是 jsonl 文件的路径
    valid_dataset = LazyJsonlDataset(
        data_path=config.valid_dataset.data_path, 
        tokenizer=tokenizer,
        max_length=config.valid_dataset.get("max_length", 2048)
    )
    # ------------------------------------------------------------------

    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # ------------------------------------------------------------------
    # 修改点：替换 Workflow 中的 Reward Function
    # ------------------------------------------------------------------
    # workflow = RLVRWorkflow(
    #     reward_fn=abcd_reward_fn,  # <--- 使用新的 reward function
    #     gconfig=config.gconfig,
    #     tokenizer=tokenizer,
    #     enable_thinking=False,
    #     dump_dir=os.path.join(
    #         StatsLogger.get_log_path(config.stats_logger), "generated"
    #     ),
    # )
    # eval_workflow = RLVRWorkflow(
    #     reward_fn=abcd_reward_fn,  # <--- 使用新的 reward function
    #     gconfig=config.gconfig.new(temperature=0.6), # Eval 时稍微降低一点温度或保持一致
    #     tokenizer=tokenizer,
    #     enable_thinking=False,
    #     rollout_stat_scope="eval-rollout",
    #     dump_dir=os.path.join(
    #         StatsLogger.get_log_path(config.stats_logger), "generated-eval"
    #     ),
    # )
    workflow = RLVRWorkflow(
        reward_fn=format_and_accuracy_reward_fn,  # <--- 替换这里
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False, # 如果你的模型输出包含 <think> 标签，这里可能需要设为 True
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    
    eval_workflow = RLVRWorkflow(
        reward_fn=format_and_accuracy_reward_fn,  # <--- 替换这里
        gconfig=config.gconfig.new(temperature=0.6), 
        tokenizer=tokenizer,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )
    # ------------------------------------------------------------------

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        with stats_tracker.record_timing("eval"):
            def evaluate_fn():
                if actor.is_data_parallel_head():
                    # 1. 设置批次大小，防止队列积压
                    eval_chunk_size = 32  
                    pending_count = 0
                    
                    for data in valid_dataloader:
                        # 假设 create_dataloader 返回的是 batch 字典
                        input_ids_batch = data["input_ids"]
                        answers_batch = data["answer"]
                        images_batch = data["image_path"]
                        
                        batch_size = len(input_ids_batch)
                        
                        for i in range(batch_size):
                            item = {
                                "input_ids": input_ids_batch[i],
                                "answer": answers_batch[i],
                                "image_path": images_batch[i]
                            }
                            eval_rollout.submit(item, eval_workflow)
                            pending_count += 1
                        
                        # 2. 如果积压请求超过阈值，强制等待并清理
                        if pending_count >= eval_chunk_size:
                            eval_rollout.wait(pending_count, timeout=None)
                            pending_count = 0
                            
                    # 3. 循环结束后，处理剩余的请求
                    if pending_count > 0:
                        eval_rollout.wait(pending_count, timeout=None)

                current_platform.synchronize()
                dist.barrier(group=actor.cpu_group)

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])