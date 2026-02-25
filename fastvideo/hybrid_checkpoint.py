import enum
import warnings
from collections import defaultdict
from functools import partial
from typing import *

import torch
import torch.nn as nn
import torch.fx.traceback as fx_traceback
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    ActivationWrapper,
    CheckpointImpl,
)
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import _CachingTorchDispatchMode, _CachedTorchDispatchMode
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import (
    SAC_IGNORED_OPS,
    CheckpointPolicy,
    SelectiveCheckpointContext,
    _is_compiling,
    _maybe_detach,
    _policy_from_bool,
    _VersionWrapper,
)
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
from torch.utils.checkpoint import noop_context_fn

class HybridCheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    The following policies are supported:

    - ``{MUST,PREFER}_SAVE``: The operation's output will be saved during the forward
      pass and will not be recomputed during the backward pass
    - ``{MUST,PREFER}_RECOMPUTE``: The operation's output will not be saved during the
      forward pass and will be recomputed during the backward pass

    Use ``MUST_*`` over ``PREFER_*`` to indicate that the policy should not be overridden
    by other subsystems like `torch.compile`.

    .. note::
        A policy function that always returns ``PREFER_RECOMPUTE`` is
        equivalent to vanilla checkpointing.

        A policy function that returns ``PREFER_SAVE`` every op is
        NOT equivalent to not using checkpointing. Using such a policy would
        save additional tensors not limited to ones that are actually needed for
        gradient computation.
    """

    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3
    MUST_SAVE_OFFLOAD = 4


HYBRID_OFFLOAD_HANDLER = None


class HybridOffloadHandler:
    def __init__(self, num_layers: int):
        self._should_offload = False
        self.current_layer = 0
        self.num_layers = num_layers

        self.storages: List[Dict[Any, List[Any]]] = []
        self.offload_infos: List[Dict[Any, List[Any]]] = []
        self.tensor_refs: List[List[int]] = []

        self.main_stream = torch.musa.current_stream()
        self.d2h_stream = torch.musa.Stream()
        self.h2d_stream = torch.musa.Stream()

    def set_offload(self, offload: bool):
        self._should_offload = offload

    def reset(self):
        self.storages = []
        self.tensor_refs = []
        self.offload_infos = []

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )

        cpu_backup.copy_(src_tensor, non_blocking=pin_memory)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None, reload_buffer=None):
        assert isinstance(state, tuple)
        device, cpu_backup = state

        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()

        if reload_buffer is None:
            return cpu_backup.to(device, non_blocking=non_blocking)

        assert (
            cpu_backup.size() == reload_buffer.size()
        ), "Can't copy two buffers of different sizes!"

        reload_buffer.copy_(cpu_backup, non_blocking=non_blocking)

        return reload_buffer

    @staticmethod
    def version_wrapper_offload(version_wrapper: _VersionWrapper):
        version_wrapper.val = HybridOffloadHandler.offload(version_wrapper.val)

    @staticmethod
    def version_wrapper_reload(version_wrapper: _VersionWrapper):
        if not isinstance(version_wrapper.val, tuple):
            return

        version_wrapper.val = HybridOffloadHandler.reload(version_wrapper.val)

    @staticmethod
    def tensor_ref_append(tensor_ref, version_wrapper: _VersionWrapper):
        tensor_ref.append(version_wrapper.val)

    def bulk_offload_layer(self, layer_to_offload: int):
        storage = self.storages[layer_to_offload]
        offload_info = self.offload_infos[layer_to_offload]

        with torch.musa.stream(self.d2h_stream):
            for func, indexes in offload_info.items():
                for index in indexes:
                    tree_map(
                        HybridOffloadHandler.version_wrapper_offload,
                        storage[func][index],
                    )

    def bulk_reload_layer(self, layer_to_reload: int):
        storage = self.storages[layer_to_reload]
        # offload_info = self.offload_infos[layer_to_reload]

        with torch.musa.stream(self.h2d_stream):
            for _, items in storage.items():
                tree_map(
                    HybridOffloadHandler.version_wrapper_reload,
                    items,
                )
                # for index in indexes:
                #     tree_map(
                #         HybridOffloadHandler.version_wrapper_reload,
                #         storage[func][index],
                #     )

    def layer_pre_forward_hook(self):
        # Runtime invariants to catch layer-index drift early.
        # The handler is a global singleton and is expected to start each iteration
        # from layer 0. If a prior backward did not traverse all wrapped layers,
        # current_layer may be non-zero here, which will misalign offload/reload.
        if not (0 <= self.current_layer <= self.num_layers):
            raise RuntimeError(
                f"[hybrid-ac] current_layer exceeded num_layers after forward: "
                f"current_layer={self.current_layer}, num_layers={self.num_layers}"
            )
        if self.current_layer > 0:
            # ensure previous layer kernel are finished
            self.d2h_stream.wait_stream(self.main_stream)
            # logger.debug(
            #     f"\033[92mstart to offload previous layer:{self.current_layer - 1} activations ...\033[0m"
            # )
            self.bulk_offload_layer(self.current_layer - 1)

    def layer_post_forward_hook(self):
        if self.current_layer > 0:
            # logger.debug(f"\033[92mstart to release previous layer:{self.current_layer - 1} activations ...\033[0m")
            self.main_stream.wait_stream(self.d2h_stream)
            # release previous layer reference tensor
            self.tensor_refs[self.current_layer - 1].clear()

        self.current_layer += 1
        if self.current_layer > self.num_layers:
            raise RuntimeError(
                f"[hybrid-ac] current_layer exceeded num_layers after forward: "
                f"current_layer={self.current_layer}, num_layers={self.num_layers}"
            )
    def layer_pre_backward_hook(self):
        self.current_layer -= 1
        # assert self.current_layer >= 0

        # next layer backward kernel finish
        self.h2d_stream.wait_stream(self.main_stream)
        # make sure previous layer h2d finished
        self.main_stream.wait_stream(self.h2d_stream)

        if self.current_layer > 0:
            # logger.debug(
            #     f"\033[33mstart to prefetch layer:{self.current_layer - 1} activations ...\033[0m"
            # )
            self.bulk_reload_layer(self.current_layer - 1)

    def layer_post_backward_hook(self):
        if self.current_layer == 0:
            self.reset()
            # for index, storage in enumerate(self.storages):
            #     for func, items in storage.items():
            #         logger.info(f"=======index:{index} =====func:{func} ========len:{len(items)}")

            # import debugpy; debugpy.breakpoint()
            # print(f"============================")


def get_hybrid_offload_handler(num_layers: int = -1): # 单例模式
    global HYBRID_OFFLOAD_HANDLER
    if num_layers <= 0:
        assert HYBRID_OFFLOAD_HANDLER is not None

    if HYBRID_OFFLOAD_HANDLER is None:
        assert num_layers > 0
        HYBRID_OFFLOAD_HANDLER = HybridOffloadHandler(num_layers)

    assert HYBRID_OFFLOAD_HANDLER.num_layers > 0
    return HYBRID_OFFLOAD_HANDLER


def set_hybrid_offload_handler(num_layers: int):
    global HYBRID_OFFLOAD_HANDLER
    assert num_layers > 0, "HYBRID_OFFLOAD_HANDLER num_layers <= 0"

    if HYBRID_OFFLOAD_HANDLER is not None:
        assert HYBRID_OFFLOAD_HANDLER.num_layers > 0
        # logger.info(f"HYBRID_OFFLOAD_HANDLER exists.")
        return

    HYBRID_OFFLOAD_HANDLER = HybridOffloadHandler(num_layers)


class _HybridCachingTorchDispatchMode(TorchDispatchMode):
    # Used together with _HybridCachedTorchDispatchMode to implement SAC.
    def __init__(self, policy_fn, storage, offload_info, tensor_ref):
        self.policy_fn = policy_fn

        self.storage = storage
        self.offload_info = offload_info
        self.tensor_ref = tensor_ref

        self.offload_handler = get_hybrid_offload_handler()
        self.offload_handler.storages.append(self.storage)
        self.offload_handler.offload_infos.append(self.offload_info)
        self.offload_handler.tensor_refs.append(self.tensor_ref)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        # is_recompute ensure we are in conventional forward
        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=False), func, *args, **kwargs
        )
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if is_compiling:
            # Overwrite each node's "recompute" tag to add in the user annotation.
            fx_traceback.current_meta["recompute"] = policy

        out = func(*args, **kwargs)

        any_ret_has_alias_info = any(
            ret.alias_info is not None for ret in func._schema.returns
        )
        # any_ret_has_alias_info = False

        if (
            policy
            in (
                HybridCheckpointPolicy.MUST_SAVE,
                HybridCheckpointPolicy.PREFER_SAVE,
                HybridCheckpointPolicy.MUST_SAVE_OFFLOAD,
            )
            or is_compiling
        ):
            if policy is HybridCheckpointPolicy.MUST_SAVE_OFFLOAD:
                self.tensor_ref.append(out)
                self.offload_info[func].append(len(self.storage[func]))

            cached_out: _VersionWrapper = tree_map(
                lambda x: _VersionWrapper(_maybe_detach(x, any_ret_has_alias_info)),
                out,
            )
            self.storage[func].append(cached_out)
            # rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0 # 经过debug cached_out应该都是tensor,没有tuple
            # if rank == 0:
            #     print(type(cached_out.val))
            #     if type(cached_out.val) == type(()):
            #         for _ in cached_out.val:
            #             print(type(_))
        
        return out


class _HybridCachedTorchDispatchMode(TorchDispatchMode):
    # Used together with _HybridCachedTorchDispatchMode to implement SAC.
    def __init__(self, policy_fn, storage, allow_cache_entry_mutation):
        self.policy_fn = policy_fn
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=True), func, *args, **kwargs
        )
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        is_compiling = _is_compiling(func, args, kwargs)

        if (
            policy
            in (
                HybridCheckpointPolicy.MUST_SAVE,
                HybridCheckpointPolicy.PREFER_SAVE,
                HybridCheckpointPolicy.MUST_SAVE_OFFLOAD,
            )
            or is_compiling
        ):
            storage = self.storage.get(func)
            if storage is None:
                raise RuntimeError(
                    f"{func} encountered during backward, but not found in storage"
                )
            if len(storage) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed to backward once "
                    "on any region computed under selective activation checkpoint."
                )
            # pop in order
            # out = tree_map(
            #     lambda x: x.get_val(self.allow_cache_entry_mutation), storage.pop(0)
            # )
            cached_wrappered_object = storage.pop(0)
            if type(cached_wrappered_object.val) != type(()):
                out = cached_wrappered_object.val
            else:
                dev, t = cached_wrappered_object.val
                out = t.to(dev)
        else:
            out = func(*args, **kwargs)
        return out


def create_hybrid_checkpoint_contexts(
    policy_fn_or_list, allow_cache_entry_mutation=False
):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.
    """
    # NB: If grad_mode is disabled, checkpoint would not run forward under
    #     context_fn anyway, so proceed as usual.
    if isinstance(policy_fn_or_list, list):
        for op in policy_fn_or_list:
            if not isinstance(op, torch._ops.OpOverload):
                _extra_msg = (
                    (
                        "Please update the OpOverloadPacket to a specific OpOverload."
                        "For example, if you have `torch.ops.aten.mm`, change it to `torch.ops.aten.mm.default`."
                    )
                    if isinstance(op, torch._ops.OpOverloadPacket)
                    else ""
                )
                raise ValueError(
                    f"Expected op in `op_list` to be an OpOverload but got: {op} "
                    f"of type {type(op)}. {_extra_msg}"
                )

        def policy_fn(ctx, op, *args, **kwargs):
            if op in policy_fn_or_list:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage = defaultdict(list)
    offload_info = defaultdict(list)
    tensor_ref = list()
    return (
        _HybridCachingTorchDispatchMode(policy_fn, storage, offload_info, tensor_ref),
        _HybridCachedTorchDispatchMode(policy_fn, storage, allow_cache_entry_mutation),
        #_CachingTorchDispatchMode(policy_fn, storage),
        #_CachedTorchDispatchMode(policy_fn, storage, allow_cache_entry_mutation)
    )


class HybridOffloadHeadModule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offload_handler: HybridOffloadHandler, *args):
        # offload_handler.set_offload()
        offload_handler.layer_pre_forward_hook()
        ctx.offload_handler = offload_handler
        return args

    @staticmethod
    def backward(ctx, *args):
        offload_handler = ctx.offload_handler
        offload_handler.layer_post_backward_hook()
        return None, *args


class HybridOffloadTailModule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offload_handler: HybridOffloadHandler, tensor):
        offload_handler.layer_post_forward_hook()
        ctx.offload_handler = offload_handler
        return tensor

    @staticmethod
    def backward(ctx, output_grad):
        offload_handler = ctx.offload_handler
        offload_handler.layer_pre_backward_hook()
        return None, output_grad


class HybridCheckpointWrapper(ActivationWrapper):
    """
    An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
        checkpoint_fn=None,
        **checkpoint_fn_kwargs,
    ):
        super().__init__(mod)
        self.checkpoint_impl = checkpoint_impl
        if checkpoint_fn is None:
            # use torch.utils.checkpoint
            self.checkpoint_fn = partial(
                torch_utils_checkpoint,
                use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                **checkpoint_fn_kwargs,
            )
        else:
            # Construct user-specified checkpoint function.
            self.checkpoint_fn = partial(
                checkpoint_fn,
                **checkpoint_fn_kwargs,
            )

        self.hybrid_handler = get_hybrid_offload_handler()

    @torch.compiler.disable
    def forward(self, *args, **kwargs):
        # step1: apply OffloadHeadModule
        #
        # IMPORTANT detach rules:
        # - Never detach the main hidden-states tensor (args[0]). Detaching it will
        #   break the autograd graph across subsequent layers.
        # - Always detach RoPE frequency tensors (freqs_cis). In this codebase,
        #   freqs_cis is passed as args[3] for both MMDoubleStreamBlock and
        #   MMSingleStreamBlock. Fused RoPE kernels are not differentiable w.r.t
        #   freq_cis, so it must not require grads.
        detachs = []
        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                detachs.append(False)
                continue
            if i == 0:
                detachs.append(False)
                continue
            if i == 3:
                detachs.append(True)
                continue
            detachs.append(False)

        internals = HybridOffloadHeadModule.apply(self.hybrid_handler, *args)
        new_inputs = []
        for arg, detach in zip(internals, detachs):
            new_inputs.append(arg.detach() if (detach and isinstance(arg, torch.Tensor)) else arg)

        # step2: apply selective checkpoint
        # Support keyword arguments for reentrant checkpoint. Note that this
        # only works if user has specified self.checkpoint_impl and is not
        # using their own custom checkpoint_fn.
        assert self.checkpoint_impl == CheckpointImpl.NO_REENTRANT

        if kwargs != {}:
            # Pack the args and kwargs
            flat_args, kwarg_keys = _pack_kwargs(*new_inputs, **kwargs)

            # Function that only takes (packed) args, but can unpack them
            # into the original args and kwargs for the checkpointed
            # function, and runs that function.
            def my_function(*inputs):
                # unpack back into args and kwargs
                unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                # run original module
                return self._checkpoint_wrapped_module(
                    *unpacked_args, **unpacked_kwargs
                )

            # Pass the function that only takes packed args into reentrant
            # checkpoint API.
            output = self.checkpoint_fn(  # type: ignore[misc]
                my_function,
                *flat_args,
            )
        else:
            output = self.checkpoint_fn(  # type: ignore[misc]
                self._checkpoint_wrapped_module, *new_inputs, **kwargs
            )

        # step3: apply OffloadTailModule (transparent hook, no gradient impact)
        current_layer = self.hybrid_handler.current_layer
        self.hybrid_handler.layer_post_forward_hook()
        
        def make_hook(offload_handler, layer_id):
            def hook(grad):
                if offload_handler.current_layer == layer_id + 1:
                    offload_handler.layer_pre_backward_hook()
                return grad
            return hook
        
        if isinstance(output, tuple):
            for t in output:
                if isinstance(t, torch.Tensor):
                    t.register_hook(make_hook(self.hybrid_handler, current_layer))
        elif isinstance(output, torch.Tensor):
            output.register_hook(make_hook(self.hybrid_handler, current_layer))
        
        return output


def hybrid_checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
    checkpoint_fn=None,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """
    Wrap a module for activation checkpointing.
    """

    if checkpoint_impl == CheckpointImpl.REENTRANT:
        warnings.warn(
            f"Please specify {CheckpointImpl.NO_REENTRANT} as "
            f"{CheckpointImpl.REENTRANT} will soon be removed as "
            "the default and eventually deprecated.",
            FutureWarning,
            stacklevel=2,
        )
    return HybridCheckpointWrapper(
        module,
        checkpoint_impl,
        checkpoint_fn,
        **checkpoint_fn_kwargs,
    )


# selective activation checkpoint operator list
_save_list = {
    #torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,  # WanAttentionBlock use linear with bias
    #torch.ops.musa.flash_attn_varlen_forward.default,
    #torch.ops.aten._scaled_dot_product_attention_flash_musa.default,
}

# selective activation offload operator list
_cpu_save_list = {
    # torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,  # WanAttentionBlock use linear with bias

    # NOTE: SDPA output is non contiguous, and the contiguous kernel may have
    # contention with other compute kernels, so we use SAC for SDPA.
    # torch.ops.aten._scaled_dot_product_attention_flash_musa.default,

    #torch.ops.musa.flash_attn_varlen_forward.default,
}


def _apply_hybrid_sac_to_transformer_block(module: nn.Module, noop=False):
    # Guard against double-wrapping the same block, which would cause
    # HybridOffloadTailModule to run twice for one logical layer (e.g. 61 vs 60).
    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            addmm_count_key = f"{mode}_addmm_count"
            if func == torch.ops.aten.addmm.default:
                meta[addmm_count_key] += 1

            to_save = func in _save_list and not (
                func == torch.ops.aten.addmm.default and (
                    meta[addmm_count_key] % 10 != 0)
            )

            if to_save:
                if func in _cpu_save_list:
                    return HybridCheckpointPolicy.MUST_SAVE_OFFLOAD
                else:
                    return HybridCheckpointPolicy.MUST_SAVE
            else:
                return HybridCheckpointPolicy.PREFER_RECOMPUTE

        return _custom_policy

    def selective_checkpointing_context_fn():
        meta = defaultdict(int)
        return create_hybrid_checkpoint_contexts(_get_custom_policy(meta))

    if noop:
        return hybrid_checkpoint_wrapper(
                module,
                context_fn=noop_context_fn,
                preserve_rng_state=False,
            )
    else:
        #return ptd_checkpoint_wrapper(
        return hybrid_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )


# def get_module_children_bottom_up(model, return_fqns: bool = False):
#     top = model if not return_fqns else ("", model)
#     stack = [top]
#     ordered_modules = []
#     while stack:
#         current_module = stack.pop()
#         if return_fqns:
#             current_module_name, current_module = current_module
#         for name, attr in current_module.named_children():
#             if isinstance(attr, torch.nn.Module):
#                 if return_fqns:
#                     child_name = current_module_name + "." + name if current_module_name else name
#                     stack.append((child_name, attr))
#                 else:
#                     stack.append(attr)
#         if return_fqns:
#             ordered_modules.append((current_module_name, current_module))
#         else:
#             ordered_modules.append(current_module)
    
#     return ordered_modules[::-1]


# def apply_ac_on_submodule(model, model_type):
#     from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
#         checkpoint_wrapper,
#     )

#     for layer_name, layer in get_module_children_bottom_up(model, True)[:-1]:
#         if len(layer_name.split(".")) > 1:
#             parent_name, child_name = layer_name.rsplit(".", 1)
#         else:
#             parent_name = None
#             child_name = layer_name

#         parent_module = model.get_submodule(parent_name) if parent_name else model

#         # allowed_module = ['self_attn', 'ffn']
#         allowed_module = ['self_attn',]
#         # WanAttentionBlock, ffn
#         # if False:
#         if isinstance(parent_module, model_type) and child_name in allowed_module:
#             # import pdb; pdb.set_trace();
#             # layer = checkpoint_wrapper(layer, preserve_rng_state=False)
#             set_hybrid_offload_handler(40)
#             layer = _apply_hybrid_ac_to_transformer_block(layer)
#             # layer = checkpoint_wrapper(layer, preserve_rng_state=False)
#             parent_module.register_module(child_name, layer)

#     return model


from collections import defaultdict
from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def _apply_selective_ac_to_transformer_block(module):
    def _get_custom_policy(meta):
        def _custom_policy(ctx, func, *args, **kwargs):
            mode = "recompute" if ctx.is_recompute else "forward"
            mm_count_key = f"{mode}_mm_count"
            if func == torch.ops.aten.addmm.default:
                meta[mm_count_key] += 1
            # to_save = func in [torch.ops.aten.addmm.default, torch.ops.aten._scaled_dot_product_attention_flash_musa.default,]
            # decide which operators to recompute.
            to_save = func in [torch.ops.aten._scaled_dot_product_attention_flash_musa.default,
                                torch.ops.aten.addmm.default,] and not (
                func == torch.ops.aten.addmm.default and meta[mm_count_key] % 10 != 0
            )
            return (
                CheckpointPolicy.MUST_SAVE
                if to_save
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _custom_policy
    def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

    return ptd_checkpoint_wrapper(
        module,
        context_fn=selective_checkpointing_context_fn,
        preserve_rng_state=False,
    )
