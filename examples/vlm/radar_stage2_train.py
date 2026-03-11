import gc
import json
import math
import os
import re
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
import wandb
from PIL import Image
from torch.utils.data import Dataset

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta

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
from areal.utils.printing import tabulate_stats
from areal.utils.stats_logger import StatsLogger
from areal.workflow.vision_rlvr import VisionRLVRWorkflow

from examples.vlm.dataset import LazyVLMJsonlDataset
from examples.vlm.reward_fn import RAdar_stage2_reward_fn
# custom_image_dir = os.environ.get("VLM_IMAGE_DIR", "/PATH/TO/YOUR/IMAGES")
custom_image_dir = os.environ.get("VLM_IMAGE_DIR", "/rice_vl/instruct/images")
# The base directory where your images are stored.
# Please update this path or set the VLM_IMAGE_DIR environment variable.

# ==========================================
# 4. Main Function
# ==========================================
def main(args):
    os.environ["SGLANG_VLM_CACHE_SIZE_MB"] = "20480" 
    # Increase SGLang VLM cache size

    config, _ = load_expr_config(args, GRPOConfig)

    if dist.is_initialized():
         pass
    else:
        dist.init_process_group("nccl")

    rank = int(os.getenv("RANK", "0"))

    if hasattr(config.train_dataset, 'num_workers'):
        config.train_dataset.num_workers = 0
    if hasattr(config.valid_dataset, 'num_workers'):
        config.valid_dataset.num_workers = 0
    seeding.set_random_seed(config.seed, f"trainer{rank}")

    # =============================
    # Weights & Biases Initialization
    # =============================
    if rank == 0:       # Only the main process writes logs to avoid conflicts.
        wandb.init(
            project="RAdaR-training",               # Project name, can be customized
            name=config.experiment_name,            # Experiment name, derived from your configuration
            config={
                "total_epochs": config.total_train_epochs,
                "batch_size": config.train_dataset.batch_size,
                "lr": config.actor.optimizer.lr,
                "kl_ctl": getattr(config.actor, 'kl_ctl', 0.0),
            },
            dir=StatsLogger.get_log_path(config.stats_logger),
            resume="allow",
        )
    else:
        wandb.init(mode="disabled")

    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    if hasattr(processor, "image_processor"):
        # Set pixel limits consistent with _smart_resize
        processor.image_processor.min_pixels = 256 * 256
        processor.image_processor.max_pixels = 1280 * 1280
    
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

    workflow = VisionRLVRWorkflow(
        reward_fn=RAdar_stage2_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    eval_workflow = VisionRLVRWorkflow(
        reward_fn=RAdar_stage2_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        processor=processor,
        enable_thinking=False,
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training
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
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                try:
                    logp = actor.compute_logp(batch)
                    batch["prox_logp"] = logp
                except ValueError as e:
                    # Catch and handle specific ValueErrors related to token mismatches during logp recomputation.
                    # These errors can occur if the generated sequence length or tokens don't match the expected shape/values
                    # during the forward pass (e.g., due to nondeterministic behavior or truncation issues).
                    # Instead of crashing, we log detailed debug info (batch IDs, error message) to help diagnose the issue.
                    if "match" in str(e) and "tokens" in str(e):
                        print(f"Error Message: {e}")
                        if 'query_id' in batch:
                            print(f"Suspect IDs in this batch: {batch['query_id']}")
                    pass
                    # raise e
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

        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)

        # Submit train stats to WandB/Logger
        train_stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        stats_logger.commit(epoch, step, global_step, train_stats)

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

        with stats_tracker.record_timing("eval"):
            def evaluate_fn():
                # Only Head handles task submission and result collection
                if actor.is_data_parallel_head():
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data: 
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                    
                # Synchronization required as export_all usually involves all_reduce
                current_platform.synchronize()
                dist.barrier(group=actor.cpu_group)

                # Export Eval stats
                eval_stats = stats_tracker.export_all(reduce_group=actor.data_parallel_group)
                # Commit Eval metrics
                stats_logger.commit(epoch, step, global_step, eval_stats)

            evaluator.evaluate(evaluate_fn, epoch, step, global_step)

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