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
from PIL import Image


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
    def __init__(self, data_path, processor, max_length=2048, base_image_path=None):
        self.data_path = data_path
        self.processor = processor
        self.max_length = max_length
        self.base_dir = os.path.dirname(data_path)
        # Use the specified base path or default to JSON file's directory
        self.base_image_path = base_image_path or self.base_dir
        self.offsets = [0]

        print(f"Indexing preprocessed dataset: {data_path} ...")
        with open(data_path, "rb") as f:
            while f.readline():
                self.offsets.append(f.tell())
        self.offsets.pop()
        print(f"Indexed {len(self.offsets)} samples.")

    def __len__(self):
        return len(self.offsets)

    def _get_image_path(self, sample):
        """Resolve image path based on base_image_path."""
        image_rel_path = sample.get("image")
        if not image_rel_path:
            return None

        if isinstance(image_rel_path, list):
            image_rel_path = image_rel_path[0]

        # Construct path using the base_image_path
        potential_path = os.path.join(self.base_image_path, image_rel_path)
        if os.path.exists(potential_path):
            return potential_path
        else:
            print(f"Warning: Image path does not exist: {potential_path}")
            return None

    def __getitem__(self, index):
        offset = self.offsets[index]

        with open(self.data_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                return self.__getitem__((index + 1) % len(self))

        # Load image object
        image_path = self._get_image_path(sample)
        image_obj = None
        if image_path:
            try:
                image_obj = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")

        # Directly get pre-formatted messages string and answer
        messages_text = sample.get("messages", "")
        answer = sample.get("answer", "")
        query_id = sample.get("query_id", str(index))

        return {
            "messages": messages_text, # Already a string
            "images": image_obj,       # PIL Image object
            "answer": answer,
            "query_id": query_id,
        }

    def _extract_ground_truth(self, sample):
        if "gt_answer" in sample:
            return str(sample["gt_answer"])
        # Fallback
        if "conversations" in sample:
            for turn in sample["conversations"]:
                if turn.get("from") in ("gpt", "assistant"):
                    return str(turn.get("value", ""))
        return ""

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

    # [Dataset] 使用自定义的 LazyVLMJsonlDataset
    # [MODIFICATION 5] Pass your custom image directory path
    custom_image_dir = "/rice_vl/instruct/images"
    # custom_image_dir = "/path/to/your/actual/image/directory" # Replace this with your actual path
    train_dataset = LazyVLMJsonlDataset(
        data_path=config.train_dataset.path,
        processor=processor,
        max_length=2048,
        base_image_path=custom_image_dir # Pass the custom path
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
                                "answer": data["answer"][i], # 答案字符串
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
   