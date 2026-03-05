# !/bin/python3
# isort: skip_file
import torch_musa
import argparse
import functools
import math
import os
import time
from collections import deque

import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from fastvideo.dataset.latent_datasets import (LatentDataset, latent_collate_function)
from fastvideo.utils.latents_utils import normalize_dit_input
from fastvideo.models.mochi_hf.pipeline_mochi import MochiPipeline
from fastvideo.models.hunyuan_hf.pipeline_hunyuan import HunyuanVideoPipeline

from fastvideo.utils.checkpoint import (resume_lora_optimizer, save_checkpoint, save_lora_checkpoint)
from fastvideo.utils.communications import (broadcast, sp_parallel_dataloader_wrapper)
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing, get_dit_fsdp_kwargs)
from fastvideo.utils.load import load_transformer
from fastvideo.utils.logging_ import main_print
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group, get_sequence_parallel_state,
                                             initialize_sequence_parallel_state)
from fastvideo.utils.te_fp8 import get_fp8_recipe, is_te_fp8_enabled, te_fp8_autocast
from fastvideo.utils.validation import log_validation
from fastvideo.version import fsdp2_supported

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.fsdp import OffloadPolicy


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size, ),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2)**2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
    return u


def get_sigmas(noise_scheduler, device, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def _parse_layer_indices(spec, total_layers):
    if spec is None:
        return set()

    spec = str(spec).strip()
    if spec == "":
        return set()

    selected = set()
    for token in spec.split(","):
        token = token.strip()
        if token == "":
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                raise ValueError(f"Invalid layer range '{token}': start must be <= end")
            for idx in range(start, end + 1):
                if idx < 0 or idx >= total_layers:
                    raise ValueError(f"Layer index out of range: {idx}, valid range is [0, {total_layers - 1}]")
                selected.add(idx)
        else:
            idx = int(token)
            if idx < 0 or idx >= total_layers:
                raise ValueError(f"Layer index out of range: {idx}, valid range is [0, {total_layers - 1}]")
            selected.add(idx)
    return selected


def _wrap_module_forward_with_te_fp8(module, fp8_recipe):
    if getattr(module, "_fastvideo_te_fp8_wrapped", False):
        return

    original_forward = module.forward

    @functools.wraps(original_forward)
    def _forward_with_te_fp8(*args, **kwargs):
        with te_fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            return original_forward(*args, **kwargs)

    module.forward = _forward_with_te_fp8
    module._fastvideo_te_fp8_wrapped = True


def _configure_te_fp8_layers(transformer, layer_spec, fp8_recipe):
    block_groups = []
    if hasattr(transformer, "double_blocks"):
        block_groups.append(transformer.double_blocks)
    if hasattr(transformer, "single_blocks"):
        block_groups.append(transformer.single_blocks)
    if len(block_groups) == 0:
        raise ValueError("Layer-wise TE FP8 requires transformer.double_blocks and/or transformer.single_blocks")

    all_blocks = []
    for blocks in block_groups:
        all_blocks.extend(list(blocks))

    total_layers = len(all_blocks)
    if total_layers == 0:
        raise ValueError("Layer-wise TE FP8 requested but no transformer blocks were found")

    selected_layers = _parse_layer_indices(layer_spec, total_layers)
    for idx in selected_layers:
        _wrap_module_forward_with_te_fp8(all_blocks[idx], fp8_recipe)

    return selected_layers, total_layers


def _replace_te_linear_with_nn_linear(module, attr_name):
    old_linear = getattr(module, attr_name, None)
    if old_linear is None:
        return False
    if isinstance(old_linear, nn.Linear):
        return False
    if not hasattr(old_linear, "weight"):
        return False

    weight = old_linear.weight
    bias = getattr(old_linear, "bias", None)
    out_features, in_features = weight.shape
    new_linear = nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(weight.detach())
        if bias is not None:
            new_linear.bias.copy_(bias.detach())
    setattr(module, attr_name, new_linear)
    return True


def _convert_non_fp8_layers_to_nn_linear(transformer, selected_layers):
    block_groups = []
    if hasattr(transformer, "double_blocks"):
        block_groups.append(transformer.double_blocks)
    if hasattr(transformer, "single_blocks"):
        block_groups.append(transformer.single_blocks)
    if len(block_groups) == 0:
        raise ValueError("Layer-wise TE FP8 requires transformer.double_blocks and/or transformer.single_blocks")

    all_blocks = []
    for blocks in block_groups:
        all_blocks.extend(list(blocks))

    replaced_linears = 0
    for idx, block in enumerate(all_blocks):
        if idx in selected_layers:
            continue
        for attr_name in ("img_attn_q", "img_attn_k", "img_attn_v", "linear_q", "linear_k", "linear_v"):
            if _replace_te_linear_with_nn_linear(block, attr_name):
                replaced_linears += 1

    return replaced_linears, len(all_blocks)

def train_one_step(
    transformer,
    model_type,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    precondition_outputs,
    max_grad_norm,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    use_te_fp8=False,
    te_fp8_format="hybrid",
    te_fp8_amax_history_len=16,
    te_fp8_amax_compute_algo="max",
    te_fp8_scaling="block",
    te_fp8_block_tile_size=128,
    te_fp8_layers="",
):
    total_loss = 0.0
    optimizer.zero_grad()
    te_fp8_enabled = is_te_fp8_enabled(model_type=model_type, explicit=use_te_fp8)
    te_fp8_recipe = get_fp8_recipe(
        fp8_format=te_fp8_format,
        amax_history_len=te_fp8_amax_history_len,
        amax_compute_algo=te_fp8_amax_compute_algo,
        scaling=te_fp8_scaling,
        block_tile_size=te_fp8_block_tile_size,
    )
    use_layerwise_te_fp8 = te_fp8_enabled and bool(str(te_fp8_layers).strip())
    for _ in range(gradient_accumulation_steps):
        (
            latents,
            encoder_hidden_states,
            latents_attention_mask,
            encoder_attention_mask,
        ) = next(loader)
        latents = normalize_dit_input(model_type, latents)
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            generator=noise_random_generator,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
        if sp_size > 1:
            # Make sure that the timesteps are the same across all sp processes.
            broadcast(timesteps)
        sigmas = get_sigmas(
            noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        with torch_musa.core.amp.autocast(dtype=torch.bfloat16):
            input_kwargs = {
                "hidden_states": noisy_model_input,
                "encoder_hidden_states": encoder_hidden_states,
                "timestep": timesteps,
                "encoder_attention_mask": encoder_attention_mask,  # B, L
                "return_dict": False,
            }
            if 'hunyuan' in model_type:
                input_kwargs["guidance"] = torch.tensor([1000.0], device=noisy_model_input.device, dtype=torch.bfloat16)
            with te_fp8_autocast(enabled=te_fp8_enabled and not use_layerwise_te_fp8, fp8_recipe=te_fp8_recipe):
                model_pred = transformer(**input_kwargs)[0]

        if precondition_outputs:
            model_pred = noisy_model_input - model_pred * sigmas
        if precondition_outputs:
            target = latents
        else:
            target = noise - latents
            
        # if dist.get_rank() == 0:
        #     tp = target.detach().float()
        #     dp = (model_pred.detach().float() - tp)

        #     print(
        #         "[target stats]",
        #         "mean:", tp.mean().item(),
        #         "std:", tp.std(unbiased=False).item(),
        #         "min:", tp.min().item(),
        #         "max:", tp.max().item(),
        #         "maxabs:", tp.abs().max().item(),
        #     )
        #     print(
        #         "[diff stats]",
        #         "mean:", dp.mean().item(),
        #         "std:", dp.std(unbiased=False).item(),
        #         "min:", dp.min().item(),
        #         "max:", dp.max().item(),
        #         "maxabs:", dp.abs().max().item(),
        #         "mse:", (dp.pow(2).mean().item()),
        #     )
        #     ts = timesteps.detach()
        #     print(
        #         "[timestep]",
        #         "min:", ts.min().item(),
        #         "max:", ts.max().item(),
        #         "mean:", ts.float().mean().item(),
        #     )
        #     sg = sigmas.detach().float()
        #     print(
        #         "[sigma]",
        #         "mean:", sg.mean().item(),
        #         "min:", sg.min().item(),
        #         "max:", sg.max().item(),
        #     )

        loss = (torch.mean((model_pred.float() - target.float())**2) / gradient_accumulation_steps)
        loss.backward()
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()

    #grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    #grad_norm = torch.tensor(0.0)
    params = [p for p in transformer.parameters() if p.grad is not None]
    if len(params) > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    else:
        grad_norm = torch.tensor(0.0)
    optimizer.step()
    lr_scheduler.step()
    return total_loss, grad_norm.item()


def main(args):
    torch.backends.mudnn.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("mccl")
    torch.musa.set_device(local_rank)
    device = torch.musa.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # If passed along, set the training seed now. On GPU...
    if args.seed is not None:
        # TODO: t within the same seq parallel group should be the same. Noise should be different.
        set_seed(args.seed + rank)
    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.

    # Create model:

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    # keep the master weight to float32
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
        use_fused_rmsnorm=args.use_fused_rmsnorm,
        use_fused_rope=args.use_fused_rope,
    )

    if args.use_lora:
        assert args.model_type != "hunyuan", "LoRA is only supported for huggingface model. Please use hunyuan_hf for lora finetuning"
        if args.model_type == "mochi":
            pipe = MochiPipeline
        elif args.model_type == "hunyuan_hf":
            pipe = HunyuanVideoPipeline
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)

    if args.resume_from_lora_checkpoint:
        lora_state_dict = pipe.lora_state_dict(args.resume_from_lora_checkpoint)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                main_print(f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                           f" {unexpected_keys}. ")

    te_fp8_enabled = is_te_fp8_enabled(model_type=args.model_type, explicit=args.use_te_fp8)
    if te_fp8_enabled and args.te_fp8_layers.strip():
        te_fp8_recipe = get_fp8_recipe(
            fp8_format=args.te_fp8_format,
            amax_history_len=args.te_fp8_amax_history_len,
            amax_compute_algo=args.te_fp8_amax_compute_algo,
            scaling=args.te_fp8_scaling,
            block_tile_size=args.te_fp8_block_tile_size,
        )
        selected_layers, total_layers = _configure_te_fp8_layers(transformer, args.te_fp8_layers, te_fp8_recipe)
        replaced_linears, _ = _convert_non_fp8_layers_to_nn_linear(transformer, selected_layers)
        main_print(
            f"--> Layer-wise TE FP8 enabled for {len(selected_layers)}/{total_layers} layers: "
            f"{sorted(selected_layers)}"
        )
        main_print(f"--> Replaced TE Linear with nn.Linear in non-FP8 layers: {replaced_linears}")

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M")
    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    if args.use_lora:
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        transformer._no_split_modules = [no_split_module.__name__ for no_split_module in no_split_modules]
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)
    # transformer = transformer.to(memory_format=torch.channels_last) # transform layout to NHWC
    if fsdp2_supported:
        # ==================== FSDP2: 创建设备网格 ====================
        # 创建一维设备网格用于数据并行
        device_mesh = init_device_mesh("musa", (world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=None,      # 绝大多数参数用BF16
            reduce_dtype=torch.float32,      # 梯度规约用FP32
            cast_forward_inputs=False,
        )
        offload_policy = None
        if args.use_cpu_offload:
            offload_policy = OffloadPolicy()
        for module_name, module in transformer.named_modules():
            if module.__class__.__name__ in [m.__name__ for m in no_split_modules]:
                fully_shard(
                    module,
                    mesh=device_mesh,           # 或使用 shard_mesh (如果是2D混合并行)
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    # reshard_after_forward=True,  # 控制前向后是否立即分片以节省内存
                )
                main_print(f"  Applied FSDP2 to submodule: {module_name}")

        # 然后包装整个模型
        fully_shard(
            transformer,
            mesh=device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
        main_print("--> model loaded and wrapped with FSDP2")
    else:
        transformer = FSDP(
            transformer,
            **fsdp_kwargs,
        )
        main_print("--> model loaded")


    if args.fsdp_prefetch_layer > 0:
        def _set_fsdp_prefetch(blocks, *, n: int, direction: str):
            assert n > 0
            if direction not in {"forward", "backward"}:
                raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

            total = len(blocks)
            for idx, block in enumerate(blocks):
                block: FSDPModule = block
                if direction == "forward":
                    # Prefetch next n layers; skip tail that doesn't have enough next layers.
                    if idx >= total - n:
                        continue
                    layers_to_prefetch = [blocks[idx + j] for j in range(1, n + 1)]
                    block.set_modules_to_forward_prefetch(layers_to_prefetch)
                else:
                    # Prefetch previous n layers; skip head that doesn't have enough previous layers.
                    if idx < n:
                        continue
                    layers_to_prefetch = [blocks[idx - j] for j in range(1, n + 1)]
                    block.set_modules_to_backward_prefetch(layers_to_prefetch)
        _set_fsdp_prefetch(transformer.double_blocks, n=args.fsdp_prefetch_layer, direction="forward")
        _set_fsdp_prefetch(transformer.single_blocks, n=args.fsdp_prefetch_layer, direction="forward")
        _set_fsdp_prefetch(transformer.double_blocks, n=args.fsdp_prefetch_layer, direction="backward")
        _set_fsdp_prefetch(transformer.single_blocks, n=args.fsdp_prefetch_layer, direction="backward")


    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules, args.selective_checkpointing)

    if args.enable_selective_ac:
        from hybrid_checkpoint import _apply_selective_ac_to_transformer_block
        for idx, block in enumerate(transformer.double_blocks):
            transformer_block = _apply_selective_ac_to_transformer_block(block)
            transformer.double_blocks[idx] = transformer_block
        for idx, block in enumerate(transformer.single_blocks):
            transformer_block = _apply_selective_ac_to_transformer_block(block)
            transformer.single_blocks[idx] = transformer_block

    if args.enable_hybrid_ac:
        #os.environ["USE_CUSTOM_VARLEN_FA"] = "1"
        # attention: you need to disable gradient checkpoint when enable_hybrid_ac
        from hybrid_checkpoint import _apply_hybrid_sac_to_transformer_block, set_hybrid_offload_handler
        set_hybrid_offload_handler(len(transformer.double_blocks) + len(transformer.single_blocks))
        for idx, block in enumerate(transformer.double_blocks):
            transformer_block = _apply_hybrid_sac_to_transformer_block(block)
            transformer.double_blocks[idx] = transformer_block
        for idx, block in enumerate(transformer.single_blocks):
            transformer_block = _apply_hybrid_sac_to_transformer_block(block)
            transformer.single_blocks[idx] = transformer_block

        # debug: check whether params are actually trainable (rank0 only)
        if rank == 0:
            try:
                p0 = next(transformer.double_blocks[0].parameters())
                main_print(
                    f"[hybrid-ac][dbg] double_blocks[0] first_param requires_grad={p0.requires_grad} "
                    f"dtype={p0.dtype} device={p0.device}"
                )
            except StopIteration:
                main_print("[hybrid-ac][dbg] double_blocks[0] has no parameters")


    # Set model as trainable.
    transformer.train()

    noise_scheduler = FlowMatchEulerDiscreteScheduler()

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    from torch_musa.optim import FusedAdamW
    optimizer = FusedAdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    init_steps = 0
    if args.resume_from_lora_checkpoint:
        transformer, optimizer, init_steps = resume_lora_optimizer(transformer, args.resume_from_lora_checkpoint,
                                                                   optimizer)
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = (LengthGroupedSampler(
        args.train_batch_size,
        rank=rank,
        world_size=world_size,
        lengths=train_dataset.lengths,
        group_frame=args.group_frame,
        group_resolution=args.group_resolution,
    ) if (args.group_frame or args.group_resolution) else DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False))

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps * args.sp_size / args.train_sp_batch_size)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if rank == 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (world_size * args.gradient_accumulation_steps / args.sp_size * args.train_sp_batch_size)
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)

    # todo future
    for i in range(init_steps):
        next(loader)
    profiling_path = os.getenv('TORCH_PROFILING_TRACE', None)
    if profiling_path is not None:
        prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.MUSA],
                schedule=torch.profiler.schedule(wait=0, warmup=5, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_path),
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
                )
    else:
        prof = None
    if prof is not None:
        prof.start()
    for step in range(init_steps + 1, args.max_train_steps + 1):
        if prof is not None:
            prof.step()
        start_time = time.perf_counter()
        loss, grad_norm = train_one_step(
            transformer,
            args.model_type,
            optimizer,
            lr_scheduler,
            loader,
            noise_scheduler,
            noise_random_generator,
            args.gradient_accumulation_steps,
            args.sp_size,
            args.precondition_outputs,
            args.max_grad_norm,
            args.weighting_scheme,
            args.logit_mean,
            args.logit_std,
            args.mode_scale,
            args.use_te_fp8,
            args.te_fp8_format,
            args.te_fp8_amax_history_len,
            args.te_fp8_amax_compute_algo,
            args.te_fp8_scaling,
            args.te_fp8_block_tile_size,
            args.te_fp8_layers,
        )

        step_time = time.perf_counter() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        # progress_bar.set_postfix({
        #     "loss": f"{loss:.4f}",
        #     "step_time": f"{step_time:.2f}s",
        #     "grad_norm": grad_norm,
        # })
        # progress_bar.update(1)
        if rank == 0:
            print(f'!!!!!! rank {rank} step {step} loss: {loss:.4f} step_time: {step_time:.2f}s grad_norm: {grad_norm:.2f}')
            wandb.log(
                {
                    "train_loss": loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "avg_step_time": avg_step_time,
                    "grad_norm": grad_norm,
                },
                step=step,
            )
        if step % args.checkpointing_steps == 0:
            if args.use_lora:
                # Save LoRA weights
                save_lora_checkpoint(transformer, optimizer, rank, args.output_dir, step, pipe)
            else:
                # Your existing checkpoint saving code
                save_checkpoint(transformer, rank, args.output_dir, step)
            dist.barrier()
        if args.log_validation and step % args.validation_steps == 0:
            log_validation(args, transformer, device, torch.bfloat16, step, shift=args.shift)

    if prof is not None:
        prof.stop()
    if args.use_lora:
        save_lora_checkpoint(transformer, optimizer, rank, args.output_dir, args.max_train_steps, pipe)
    else:
        save_checkpoint(transformer, rank, args.output_dir, args.max_train_steps)

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",
                        type=str,
                        default="mochi",
                        help="The type of model to train. Currentlt support [mochi, hunyuan_hf, hunyuan]")
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--uncond_prompt_dir", type=str)
    parser.add_argument(
        "--validation_sampling_steps",
        type=str,
        default="64",
        help="use ',' to split multi sampling steps",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=str,
        default="4.5",
        help="use ',' to split multi scale",
    )
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
              " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
              " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--shift", type=float, default=1.0, help=("Set shift to 7 for hunyuan model."))
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous checkpoint. Use a path saved by"
              ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
              ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),
    )


    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
              " https://pytorch.org/docs/stable/notes/musa.html#tensorfloat-32-tf32-on-ampere-devices"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )
    parser.add_argument(
        "--enable_selective_ac",
        action="store_true",
        help="enable selective activation checkpoint",
    )
    parser.add_argument(
        "--enable_hybrid_ac",
        action="store_true",
        help="enable hybrid selective activation checkpoint && activation offload",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_rank", type=int, default=128, help="LoRA rank parameter. ")
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
              ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to apply.")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument("--use_te_fp8", action="store_true", help="Enable TransformerEngine FP8 autocast.")
    parser.add_argument(
        "--te_fp8_format",
        type=str,
        default="hybrid",
        choices=["hybrid", "e4m3"],
        help="TransformerEngine FP8 format.",
    )
    parser.add_argument(
        "--te_fp8_amax_history_len",
        type=int,
        default=16,
        help="TransformerEngine FP8 amax history length.",
    )
    parser.add_argument(
        "--te_fp8_amax_compute_algo",
        type=str,
        default="max",
        choices=["max", "most_recent"],
        help="TransformerEngine FP8 amax compute algorithm.",
    )
    parser.add_argument(
        "--te_fp8_scaling",
        type=str,
        default="block",
        choices=["block", "tensor"],
        help="TransformerEngine FP8 scaling mode.",
    )
    parser.add_argument(
        "--te_fp8_block_tile_size",
        type=int,
        default=128,
        help="TransformerEngine FP8 block scaling tile size.",
    )
    parser.add_argument(
        "--te_fp8_layers",
        type=str,
        default="",
        help="Comma-separated FP8 layer indices/ranges over [double_blocks + single_blocks], e.g. '1-58,63'.",
    )
    parser.add_argument(
        "--use_fused_rmsnorm",
        action="store_true",
        default=False,
        help="use torch fused rmsnorm implementation",
    )
    parser.add_argument(
        "--use_fused_rope",
        action="store_true",
        default=False,
        help="use torch fused rope implementation",
    )
    parser.add_argument(
        "--fsdp_prefetch_layer",
        type=int,
        default=0,
        help="fsdp prefetch argument in bwd and fwd stages"
    )

    args = parser.parse_args()
    main(args)
