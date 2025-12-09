import os
import re
import sys
import json
import gc
from copy import deepcopy
from typing import List, Dict, Any

import wandb
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
# from areal.dataset import get_custom_dataset # 替换为自定义 Lazy Dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.vision_rlvr import VisionRLVRWorkflow


# ==========================================
# 1. 辅助工具函数
# ==========================================
import re

def _normalize_text(text: str) -> str:
    """去除空白字符并转小写"""
    return "".join(str(text).lower().split())

def step1_extract_answer_from_answer_tag(text: str) -> str:
    """
    从文本中提取 </answer> 之前的内容作为答案。
    要求答案格式为：...答案内容...</answer>
    如果匹配不到 </answer>，则返回空字符串。
    """
    # 尝试匹配 ...答案...</answer>，提取 </answer> 之前的所有内容（贪婪匹配最后一个）
    # 使用 r'(.*)</answer>' 并从右向左找更可靠
    match = re.search(r'(.*)</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""  # 无 </answer> 标签，视为格式错误

# ==========================================
# 2. 复合 Reward Function (格式 + 正确性)
# ==========================================

def step1_format_and_accuracy_reward_fn(prompt, completions, answer, **kwargs):
    """
    格式要求: completion 必须包含 </answer>，且答案写在 </answer> 之前。
    """
    R_FORMAT = 0.3
    R_CORRECT = 0.7
    
    completion_text = completions
    current_reward = 0.0

    # 1. Format Check: 是否包含 </answer>（不区分大小写）
    if re.search(r'</answer>', completion_text, re.IGNORECASE):
        current_reward += R_FORMAT


    # 2. Accuracy Check
    pred_answer = step1_extract_answer_from_answer_tag(completion_text)
    pred_norm = _normalize_text(pred_answer)
    answer = step1_extract_answer_from_answer_tag(answer)
    gt_norm = _normalize_text(answer)

    if pred_norm and gt_norm:
        # 精确匹配
        if pred_norm == gt_norm:
            current_reward += R_CORRECT
        # 选择题容错（例如 GT="D"，预测为 "D" 或 "option D" 等）
        elif len(gt_norm) == 1 and gt_norm in pred_norm:
            # 限制宽松匹配的范围，避免误判
            if pred_norm in [gt_norm, f"option{gt_norm}", f"choice{gt_norm}", f"answer{gt_norm}", f"answer:{gt_norm}"]:
                current_reward += R_CORRECT
    print(f"Pred Answer: {repr(pred_answer)}, GT Answer: {repr(answer)}, Reward: {current_reward}")
    return current_reward
# ==========================================
# 3. VLM Lazy Dataset (防 OOM 关键)
# ==========================================

class LazyVLMJsonlDataset(Dataset):
    def __init__(self, data_path, processor, max_length=2048, base_image_path=None, print_example=True):
        self.data_path = data_path
        self.processor = processor # 这里的 processor 通常包含 tokenizer
        self.max_length = max_length
        self.base_dir = os.path.dirname(data_path)
        # [MODIFICATION 1] Store the custom base path
        self.base_image_path = base_image_path
        self.offsets = [0]
        
        print(f"Indexing dataset (Lazy Loading): {data_path} ...")
        # 建立文件索引，不读取内容
        with open(data_path, "rb") as f:
            while f.readline():
                self.offsets.append(f.tell())
        self.offsets.pop() # 移除最后 EOF 的 offset
        print(f"Indexed {len(self.offsets)} samples.")

        # [NEW] 打印并保存示例
        if print_example and len(self.offsets) > 0:
            self._print_and_save_example()

    def __len__(self):
        return len(self.offsets)

    def _get_image_path(self, sample):
        """解析图片路径，优先使用 base_image_path"""
        # in vision_rlvr.py
        if sample.get("image") is None:
            if sample.get("tos_key") is None:
                raise ValueError(f"Sample {sample.get('id', 'unknown')} has no images!")
            else:
                image_rel_path = sample["tos_key"][0]
        else:
            image_rel_path = sample["image"]

        if not image_rel_path:
            raise ValueError(f"Sample {sample.get('idx', 'unknown')} has no images!")

        if isinstance(image_rel_path, list):
            image_rel_path = image_rel_path[0]
        
        # [MODIFICATION 2] Check if base_image_path is set
        if self.base_image_path:
            # Construct path using the custom base path
            potential_path = os.path.join(self.base_image_path, image_rel_path)
            if os.path.exists(potential_path):
                return potential_path
            else:
                print(f"Warning: Image path constructed from base_image_path does not exist: {potential_path}")
                return None
        else:
            # Fall back to old logic if no base_image_path is provided
            if not os.path.isabs(image_rel_path):
                 potential_path = os.path.join(self.base_dir, image_rel_path)
                 if os.path.exists(potential_path):
                     return potential_path
                 else:
                     # 尝试相对于当前工作目录
                     cwd_path = os.path.abspath(image_rel_path)
                     if os.path.exists(cwd_path):
                         image_path = cwd_path
                         return image_path
            elif os.path.exists(image_rel_path): # If it was already an absolute path in the JSON
                return image_rel_path
            
        return None # Return None if no path is found

    def _build_messages(self, sample):
        """
        构建输入模型的完整 prompt 字符串。
        - 仅将原始用户问题（含 <image> 占位符）传入 chat_template；
        - 将 'append' 和 'prompt_suffix' 拼接到 template 输出的末尾；
        - 确保最终格式为：...Question...\nChoices:...\n<|assistant|>...<think_off></think_off><answer>You must give... \boxed{...}
        """
        # prompt_suffix = "You must give the answer in this form finally: \\boxed{your_answer}."
        user_content = ""
        user_append = ""

        # 1. 从 sample 中提取 user 的 content 和 append
        if "messages" in sample:
            for message in sample["messages"]:
                if message.get("role") == "user":
                    val = message.get("content", "")
                    if isinstance(val, (str, list)):
                        user_content = val
                    user_append = message.get("append", "")
                    break  # 只取第一个 user 消息

        # 2. 构建纯净的 messages（不包含 append 和 suffix）
        messages = []
        if isinstance(user_content, str):
            # 处理含 <image> 的字符串
            if "<image>" in user_content:
                parts = user_content.split("<image>")
                content_parts = []
                for i, part in enumerate(parts):
                    if part:
                        content_parts.append({"type": "text", "text": part})
                    if i < len(parts) - 1:
                        content_parts.append({"type": "image"})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": user_content}]})
        elif isinstance(user_content, list):
            # 已是多模态格式（如 [{"type": "text", ...}, {"type": "image"}]）
            messages.append({"role": "user", "content": user_content})
        else:
            # 兜底：转为字符串
            messages.append({"role": "user", "content": [{"type": "text", "text": str(user_content)}]})

        # 3. 获取 tokenizer 并应用 chat template
        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                # 关键：只传入纯净 messages，不包含 append/suffix
                base_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # 添加助手开始生成的前缀（如 <|start_header_id|>assistant<|end_header_id|>\n\n）
                )
            except Exception as e:
                print(f"Warning: apply_chat_template failed: {e}. Falling back to plain text.")
                # 回退：仅拼接文本部分
                base_text = ""
                for part in messages[0]["content"]:
                    if part.get("type") == "text":
                        base_text += part["text"]
                base_prompt = base_text
        else:
            print("Warning: Tokenizer has no 'apply_chat_template'. Using plain text fallback.")
            base_text = ""
            for part in messages[0]["content"]:
                if part.get("type") == "text":
                    base_text += part["text"]
            base_prompt = base_text

        # 4. ✅ 关键修正：在 template 输出后拼接 append
        final_prompt = base_prompt + user_append
        print(f"Final prompt after applying template and appending:\n{final_prompt}\n{'-'*40}")
        return final_prompt


    def _extract_ground_truth(self, sample):
        # [MODIFICATION] 从 gt_answer 字段提取，并保留原始格式（包括标签）
        # 训练时可能需要原始格式，奖励函数会处理提取
        raw_answer = sample.get("gt_answer", "")
        # 如果 gt_answer 为空，尝试从 assistant 消息中提取
        if not raw_answer:
            raise ValueError(f"Sample {sample.get('idx', 'unknown')} has no gt_answer!")

        return raw_answer
    
    def _smart_resize(self, image_obj, factor=28):
        W, H = image_obj.size
        
        min_pixels = 256 * 256
        max_pixels = 1280 * 1280
        
        if W * H > max_pixels:
            scale = (max_pixels / (W * H)) ** 0.5
            W = int(W * scale)
            H = int(H * scale)
        elif W * H < min_pixels:
            scale = (min_pixels / (W * H)) ** 0.5
            W = int(W * scale)
            H = int(H * scale)
            
        W = int(round(W / factor) * factor)
        H = int(round(H / factor) * factor)
        
        W = max(factor, W)
        H = max(factor, H)
        
        if W != image_obj.size[0] or H != image_obj.size[1]:
            # [FIX] 检查是否存在 Resampling 属性 (Pillow >= 9.0)
            if hasattr(Image, "Resampling"):
                resample_method = Image.Resampling.LANCZOS
            else:
                # 旧版 Pillow 使用 Image.LANCZOS
                resample_method = Image.LANCZOS
            
            image_obj = image_obj.resize((W, H), resample_method)
            
        return image_obj

    def __getitem__(self, index):
        offset = self.offsets[index]
        
        # 懒加载：此时才读取文件
        with open(self.data_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                # 遇到坏数据，递归读取下一个
                return self.__getitem__((index + 1) % len(self))

        image_path = self._get_image_path(sample)
        # [MODIFICATION 3] Use the updated _build_messages to get the formatted string
        messages_text = self._build_messages(sample)

        image_obj = None
        if image_path:
            from PIL import Image
            try:
                image_obj = Image.open(image_path).convert('RGB')
            except Exception as e:
                # 1. 如果打开失败，打印警告
                print(f"Warning: Could not load image {image_path}: {e}. Skipping to next sample...")
                # 2. 【核心逻辑】递归调用，读取下一个样本 (index + 1)
                # 取模是为了防止 index 超出范围
                return self.__getitem__((index + 1) % len(self))

            # image_obj = self._smart_resize(image_obj, factor=28)

        # [MODIFICATION] 提取 ground truth answer
        # 注意：这里 raw_answer 是包含标签的原始字符串
        raw_answer = self._extract_ground_truth(sample)
    
        return {
            "messages": messages_text, # This is now a STRING after apply_chat_template, including appended content
            "images": image_obj,       # This is a PIL Image object
            "answer": raw_answer, 
            # "input_ids": torch.tensor(input_ids, dtype=torch.long), # 不再需要 input_ids
            # 保留 query_id 以便 dump 时使用
            "query_id": sample.get("query_id", sample.get("id", sample.get("qid", str(index))))
        }

    def _print_and_save_example(self):
        """打印并保存一个示例，展示数据处理流程"""
        print("\n--- Dataset Example ---")
        # 读取第一个样本
        with open(self.data_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        try:
            sample = json.loads(first_line)
        except json.JSONDecodeError:
            print("Could not parse the first line for example.")
            return

        print(f"Raw sample keys: {list(sample.keys())}")
        print(f"Raw 'messages': {sample.get('messages', 'N/A')}")
        print(f"Raw 'gt_answer': {repr(sample.get('gt_answer', 'N/A'))}")

        # 模拟 _build_messages 过程
        messages_text = self._build_messages(sample)
        raw_answer = self._extract_ground_truth(sample)
        
        print(f"\nProcessed 'messages' (after applying template and adding append):")
        print(repr(messages_text))
        print(f"\nRaw GT Answer: {repr(raw_answer)}")
            
        # 保存示例到文件
        example_output_path = "dataset_example_output.txt"
        with open(example_output_path, "a", encoding="utf-8") as f:
            f.write("--- Dataset Example ---\n")
            f.write(f"Raw sample keys: {list(sample.keys())}\n")
            f.write(f"Raw 'messages': {sample.get('messages', 'N/A')}\n")
            f.write(f"Raw 'gt_answer': {repr(sample.get('gt_answer', 'N/A'))}\n\n")
            f.write("Processed 'messages' (after applying template and adding append):\n")
            f.write(messages_text)
            f.write("\n\n")
            f.write(f"Raw GT Answer: {repr(raw_answer)}\n")
        print(f"\nExample also saved to: {example_output_path}")
        print("--- End Example ---\n")


# ==========================================
# 4. Main Function
# ==========================================

def main(args):
    which_step = 1  # 仅支持 step 1
    config, _ = load_expr_config(args, GRPOConfig)
    rank = int(os.getenv("RANK", "0"))

    # [OOM Fix 1] 强制禁用多进程加载
    if hasattr(config.train_dataset, 'num_workers'):
        config.train_dataset.num_workers = 0
    if hasattr(config.valid_dataset, 'num_workers'):
        config.valid_dataset.num_workers = 0

    seeding.set_random_seed(config.seed, f"trainer{rank}")
    # =============================
    # Weights & Biases 初始化
    # =============================
    if rank == 0:
        wandb.init(
            project="areal-vlm-training",  # 项目名称，可自定义
            name=config.experiment_name,         # 实验名称，来自你的配置
            config={
                "total_epochs": config.total_train_epochs,
                "batch_size": config.train_dataset.batch_size,
                "lr": config.actor.optimizer.lr,
                "kl_ctl": getattr(config.actor, 'kl_ctl', 0.0),
            },
            dir=StatsLogger.get_log_path(config.stats_logger),
            resume="allow",
        )

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    # if hasattr(processor, "image_processor"):
    #     # 1. 设置与 _smart_resize 一致的像素限制
    #     processor.image_processor.min_pixels = 256 * 256
    #     processor.image_processor.max_pixels = 1280 * 1280
        
        # 2. 强制关闭 Processor 的自动 Resize 功能
        # 因为我们在 Dataset 里已经手动 resize 到完美尺寸了，不需要 Processor 再插手
        # if hasattr(processor.image_processor, "do_resize"):
        #     processor.image_processor.do_resize = False
        #     print(f"Processor config updated: do_resize={getattr(processor.image_processor, 'do_resize', 'Unknown')}")

    # CUSTOM_SPECIAL_TOKENS = [
    #         "<think_on>",
    #         "</think_on>",
    #         "<think_off>",
    #         "</think_off>",
    #         "<answer>",
    #         "</answer>"
    #     ]

    # # 添加为 special tokens（关键：set special=True）
    # tokenizer.add_tokens(CUSTOM_SPECIAL_TOKENS, special_tokens=True)

    # for tok in CUSTOM_SPECIAL_TOKENS:
    #     ids = tokenizer.encode(tok, add_special_tokens=False)
    #     print(f"{tok} -> {ids} (length: {len(ids)})")

    # [Dataset] 使用自定义的 LazyVLMJsonlDataset
    custom_image_dir = "/rice_vl/instruct/images"
    
    train_dataset = LazyVLMJsonlDataset(
        data_path=config.train_dataset.path,
        processor=processor,
        max_length=2048,
        base_image_path=custom_image_dir, # Pass the custom path
        print_example=(rank == 0) # Only print example on rank 0
    )
    valid_dataset = LazyVLMJsonlDataset(
        data_path=config.valid_dataset.path,
        processor=processor,
        max_length=2048,
        base_image_path=custom_image_dir # Pass the custom path
    )

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

    # Initialize inference engine (Remote SGLang)
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
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

    if which_step == 1:
        reward_fn = step1_format_and_accuracy_reward_fn
    elif which_step == 2:
        raise NotImplementedError("Step 2 reward function is not implemented yet.")
    else:
        raise ValueError("This script only supports step 1 training.")

    # [Workflow] 使用 VisionRLVRWorkflow 和自定义 Reward
    workflow = VisionRLVRWorkflow(
        reward_fn=reward_fn, # 你的自定义 Reward
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        enable_thinking=False, # 根据模型是否支持 think 标签调整
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = VisionRLVRWorkflow(
        reward_fn=reward_fn, # 你的自定义 Reward
        gconfig=config.gconfig.new(temperature=0.6), # Eval 温度
        tokenizer=tokenizer,
        processor=processor,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

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

    # ==========================
    # Train Loop
    # ==========================
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
            # 如果显存依然不够，尝试调小 config.actor.group_size (GRPO采样的数量)
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                # ================= [修改开始] =================
                try:
                    logp = actor.compute_logp(batch)
                except ValueError as e:
                    # 专门捕获 tokens 不匹配的错误
                    if "match" in str(e) and "tokens" in str(e):
                        print("\n" + "="*50)
                        print("!!! 捕获到导致 Crash 的 Bad Batch !!!")
                        print(f"Error Message: {e}")
                        
                        # 尝试从 batch 中提取 ID 信息
                        # 注意：这取决于你的 collate_fn 是如何打包数据的
                        # 通常 batch 是一个字典，里面可能有 'query_id', 'id', 或者 'meta_info'
                        if 'query_id' in batch:
                            print(f"Suspect IDs in this batch: {batch['query_id']}")
                        elif 'id' in batch:
                            print(f"Suspect IDs in this batch: {batch['id']}")
                        else:
                            print("Batch keys available:", batch.keys())
                            # 如果没有 ID，尝试打印 input_ids 的形状或者其他元数据
                        
                        print("="*50 + "\n")
                    raise e # 打印完后继续抛出异常，中断程序
                    # ================= [修改结束] =================
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
            saver.save(
                actor,
                epoch,
                step,
                global_step,
                tokenizer=tokenizer,
                processor=processor,
            )

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
                processor=processor,
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # [OOM Fix 4] 评估阶段分块提交 (Chunked Evaluation)
        with stats_tracker.record_timing("eval"):
            def evaluate_fn():
                if actor.is_data_parallel_head():
                    chunk_size = 32  # 每次只处理 32 个样本
                    pending_cnt = 0
                    
                    # 遍历 dataloader
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            pending_cnt += 1
                            
                        # 下面的逻辑保持不变
                        if pending_cnt >= chunk_size:
                            eval_rollout.wait(pending_cnt, timeout=None)
                            pending_cnt = 0
                            gc.collect() 

                    if pending_cnt > 0:
                        eval_rollout.wait(pending_cnt, timeout=None)

                current_platform.synchronize()
                dist.barrier(group=actor.cpu_group)

            evaluator.evaluate(evaluate_fn, epoch, step, global_step)

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, stats)

        # =============================
        # Weights & Biases 日志记录
        # =============================
        if rank == 0:
            log_dict = {}

            # 1. 从 stats 中提取你截图中的所有指标
            # 注意：AReaL 的 stats 是扁平化的 key-value，键名与你截图一致
            keys_to_log = [
                # Final Reward
                "ppo_actor/final_reward/avg",
                # Task Reward (你的 reward_fn 输出)
                "ppo_actor/task_reward/avg",
                # Actor Loss
                "ppo_actor/update/actor_loss/avg",
                # Advantages
                "ppo_actor/advantages/avg",
                # Sequence Length
                "ppo_actor/seq_len/avg",
                # Prompt Length
                "ppo_actor/prompt_len/avg",
                # KL / Entropy / Clip
                "ppo_actor/kl_mean",
                "ppo_actor/entropy_mean",
                "ppo_actor/eps_clip",
                "ppo_actor/clip_fraction",

                # Update Stats
                "ppo_actor/update/update_successful",
                "ppo_actor/update/unclipped_behave_tokens",
                "ppo_actor/update/vocab_max_logits/avg",
                "ppo_actor/update/vocab_min_logits/avg",

                # Other useful metrics
                "ppo_actor/correct_n_seqs",
                "ppo_actor/incorrect_n_seqs",
                "ppo_actor/incorrect_seq_len/avg",
                "ppo_actor/no_eos_ratios/max",
                "ppo_actor/old_logp/min",
                "ppo_actor/old_logp/max",
            ]

            for key in keys_to_log:
                # 使用 .get 避免 KeyError
                log_dict[key] = stats.get(key, 0.0)

            # # 记录评估结果（如果 evaluator 支持）
            # if hasattr(evaluator, 'last_eval_result') and evaluator.last_eval_result:
            #     eval_res = evaluator.last_eval_result
            #     log_dict.update({
            #         "eval/reward_mean": eval_res.get("reward_mean", 0.0),
            #     })
            wandb.log(log_dict, step=global_step)

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