def main(args):
    config, _ = load_expr_config(args, GRPOConfig)

    rank = int(os.getenv("RANK", "0"))

    # [Fix] 禁用 DataLoader 多进程（避免 fork + CUDA 冲突或内存碎片）
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

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    # [VLM] 自定义图像根目录
    custom_image_dir = "/rice_vl/instruct/images"

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        processor=processor,
        base_image_path=custom_image_dir,  # 传递自定义路径
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        processor=processor,
        base_image_path=custom_image_dir,
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

    # Create rollout workflow
    workflow = VisionRLVRWorkflow(
        reward_fn=format_and_accuracy_reward_fn,  # 使用你自定义的 reward
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        enable_thinking=False,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = VisionRLVRWorkflow(
        reward_fn=format_and_accuracy_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),  # Eval 用不同温度
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

        # pause inference for updating weights, save, and evaluation
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

        with stats_tracker.record_timing("eval"):
            def evaluate_fn():
                if actor.is_data_parallel_head():
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
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