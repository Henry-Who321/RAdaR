import os
import re
import sys
import json
import gc
from copy import deepcopy
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

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

def _normalize_text(text: str) -> str:
    """去除空白字符并转小写"""
    return "".join(str(text).lower().split())

def _extract_answer_from_box(text: str) -> str:
    """提取 \\boxed{} 中的内容，增强鲁棒性"""
    # 移除 think 标签
    text_wo_think = re.sub(r"<\|think\|>.*?</\|think\|>", "", text, flags=re.DOTALL)
    text_wo_think = re.sub(r"<think>.*?</think>", "", text_wo_think, flags=re.DOTALL | re.IGNORECASE)

    # 优先匹配 standard latex boxed
    boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    m = re.search(boxed_pattern, text_wo_think)
    if m:
        return m.group(1).strip()
    
    # 备选匹配
    m = re.search(r"<\|answer\|>(.*?)</\|answer\|>", text_wo_think, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    m = re.search(r"<answer>(.*?)</answer>", text_wo_think, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 如果没有格式，清理标签作为保底
    no_tags = re.sub(r"</?[^>]+>", "", text_wo_think)
    return no_tags.strip()

# ==========================================
# 2. 复合 Reward Function (格式 + 正确性)
# ==========================================

def format_and_accuracy_reward_fn(prompt, completions, answer, **kwargs):
    """
    Reward = Format Reward (0.4) + Accuracy Reward (1.0)
    """
    R_FORMAT = 0.4
    R_CORRECT = 1.0 
    
    # completions 是字符串，直接使用 (VisionRLVRWorkflow._compute_rewards 传入)
    completion_text = completions
    
    current_reward = 0.0
    
    # 1. Format Check
    if "\\boxed{" in completion_text:
        current_reward += R_FORMAT
        
    # 2. Accuracy Check
    pred_answer = _extract_answer_from_box(completion_text)
    pred_norm = _normalize_text(pred_answer)
    gt_norm = _normalize_text(answer)
    
    if pred_norm and gt_norm:
        if pred_norm == gt_norm:
            current_reward += R_CORRECT
        # 选择题容错: GT="D", Pred="OptionD" / "Answer:D"
        elif len(gt_norm) == 1 and gt_norm in pred_norm:
             # 简单的包含检查，防止 loose match
             if pred_norm in [gt_norm, f"option{gt_norm}", f"choice{gt_norm}", f"answer{gt_norm}"]:
                 current_reward += R_CORRECT

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
        
        # 获取 Tokenizer (可能不需要，因为 messages_text 已经格式化)
        # tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        # input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        # if len(input_ids) > self.max_length:
        #     input_ids = input_ids[-self.max_length:]

        # Load image object if path exists
        image_obj = None
        if image_path:
            from PIL import Image
            try:
                image_obj = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
                # 如果加载失败，仍然保留 None

        # [MODIFICATION] 提取 ground truth answer
        # 注意：这里 raw_answer 是包含标签的原始字符串
        raw_answer = self._extract_ground_truth(sample)
        # 清理 answer (用于奖励函数比较) - 这部分可以保留在 dataset 里，也可以移至 reward_fn
        clean_answer = _extract_answer_from_box(raw_answer)
        if not clean_answer:
            clean_answer = raw_answer.strip() # 如果提取失败，使用原始清理版

        # [MODIFICATION 4] 返回 item 格式，符合 VisionRLVRWorkflow.arun_episode 的期望
        # VisionRLVRWorkflow.arun_episode 期望 data 参数包含 "images", "messages", "answer"
        # "messages" 现在是格式化后的字符串
        # "images" 是 PIL Image 对象
        return {
            "messages": messages_text, # This is now a STRING after apply_chat_template, including appended content
            "images": image_obj,      # This is a PIL Image object
            "answer": raw_answer, # 使用原始包含标签的 answer 进行训练和奖励计算
            # "input_ids": torch.tensor(input_ids, dtype=torch.long), # 不再需要 input_ids
            # "image_path": image_path, # 不再需要路径，因为图像对象已加载
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
        clean_answer = _extract_answer_from_box(raw_answer)
        
        print(f"\nProcessed 'messages' (after applying template and adding append):")
        print(repr(messages_text))
        print(f"\nRaw GT Answer: {repr(raw_answer)}")
        print(f"Clean GT Answer (from reward fn): {repr(clean_answer)}")
            
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
            f.write(f"Clean GT Answer (from reward fn): {repr(clean_answer)}\n")
        print(f"\nExample also saved to: {example_output_path}")
        print("--- End Example ---\n")


# ==========================================
# 4. Main Function
# ==========================================

def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    rank = int(os.getenv("RANK", "0"))

    # [OOM Fix 1] 强制禁用多进程加载
    if hasattr(config.train_dataset, 'num_workers'):
        config.train_dataset.num_workers = 0
    if hasattr(config.valid_dataset, 'num_workers'):
        config.valid_dataset.num_workers = 0

    seeding.set_random_seed(config.seed, f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # VLM 必须加载 Processor (含 Tokenizer)
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

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
    # [MODIFICATION 5] Pass your custom image directory path
    custom_image_dir = "/rice_vl/instruct/images"
    # custom_image_dir = "/path/to/your/actual/image/directory" # Replace this with your actual path
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

    # [Workflow] 使用 VisionRLVRWorkflow 和自定义 Reward
    workflow = VisionRLVRWorkflow(
        reward_fn=format_and_accuracy_reward_fn, # 你的自定义 Reward
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        enable_thinking=False, # 根据模型是否支持 think 标签调整
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = VisionRLVRWorkflow(
        reward_fn=format_and_accuracy_reward_fn,
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
        actor, saver, evaluator, stats_logger, train_dataloader,
        inference_engine=rollout, weight_update_meta=weight_update_meta,
    )
    start_step = recover_info.last_step_info.next().global_step if recover_info else 0

    max_steps = config.total_train_epochs * len(train_dataloader)

    # ==========================
    # Train Loop
    # ==========================
    for global_step in range(start_step, max_steps):
        # [OOM Fix 2] 主动垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

        epoch = global_step // len(train_dataloader)
        step = global_step % len(train_dataloader)
        step_info = StepInfo(global_step=global_step, epoch=epoch, epoch_step=step, steps_per_epoch=len(train_dataloader))

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

        # [OOM Fix 3] 删除 batch 释放显存
        del batch

        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer, processor=processor)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor, step_info, saver, evaluator, stats_logger, 
                train_dataloader, tokenizer=tokenizer, processor=processor
            )

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # [OOM Fix 4] 评估阶段分块提交 (Chunked Evaluation)
        with stats_tracker.record_timing("eval"):
            def evaluate_fn():
                if actor.is_data_parallel_head():
                    chunk_size = 32  # 每次只处理 32 个样本
                    pending_cnt = 0
                    
                    # 遍历 dataloader (此时 LazyDataset 只读取需要的行)
                    for data in valid_dataloader:
                        # dataloader 会将 dataset 的单个样本合并成 batch
                        # 因此 data 是一个字典，其键是 dataset.__getitem__ 返回的键，其值是 stacked 的 tensor 或 list
                        # 例如 data["images"] 可能是一个 PIL Image 对象的列表
                        batch_sz = len(data["images"]) # 或者 len(data["answer"])
                        
                        for i in range(batch_sz):
                            item = {
                                # 从 batch 中取出第 i 个样本的数据
                                "images": data["images"][i], # PIL Image 对象
                                "messages": data["messages"][i], # 格式化后的字符串
                                "answer": data["answer"][i], # 答案字符串 (原始带标签)
                                # 如果 dataset 中包含了 query_id
                                "query_id": data.get("query_id", [str(i) for i in range(batch_sz)])[i],
                            }
                            eval_rollout.submit(item, eval_workflow)
                            pending_cnt += 1
                        
                        # 如果积压太多，强制等待 SGLang 处理完
                        if pending_cnt >= chunk_size:
                            eval_rollout.wait(pending_cnt, timeout=None)
                            pending_cnt = 0
                            gc.collect() # 清理 Python 对象

                    # 处理剩余尾部
                    if pending_cnt > 0:
                        eval_rollout.wait(pending_cnt, timeout=None)
                
                current_platform.synchronize()
                dist.barrier(group=actor.cpu_group)

            evaluator.evaluate(evaluate_fn, epoch, step, global_step)

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