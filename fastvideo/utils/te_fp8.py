import os
from contextlib import contextmanager

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
except Exception:
    te = None
    recipe = None


def is_te_fp8_enabled(model_type=None, explicit=None, precision=None):
    if explicit is not None:
        enabled = bool(explicit)
    elif precision is not None:
        enabled = str(precision).lower() == "fp8"
    else:
        enabled = str(os.getenv("FASTVIDEO_USE_TE_FP8", "0")).lower() in {"1", "true", "yes", "on"}

    if model_type is not None and "hunyuan" not in str(model_type).lower():
        return False

    return enabled and te is not None and recipe is not None


def get_fp8_recipe(
    fp8_format="hybrid",
    amax_history_len=16,
    amax_compute_algo="max",
    scaling="block",
    block_tile_size=128,
):
    if recipe is None:
        return None

    fmt = recipe.Format.HYBRID if str(fp8_format).lower() == "hybrid" else recipe.Format.E4M3
    mode = str(scaling).lower()
    if mode == "block":
        return recipe.MXFP8BlockScaling(0, fmt, False, False, block_tile_size)
    if mode == "tensor":
        return recipe.DelayedScaling(
            fp8_format=fmt,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
        )

    raise ValueError(f"Unsupported te_fp8 scaling mode: {scaling}. Only 'block' and 'tensor' are allowed.")


@contextmanager
def te_fp8_autocast(enabled=False, fp8_recipe=None):
    if not enabled or te is None:
        yield
        return

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        yield
