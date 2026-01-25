import logging
import os


def init_tensorboard(cfg, output_dir, local_rank):
    if not hasattr(cfg, "TENSORBOARD") or not cfg.TENSORBOARD.ENABLED:
        return None
    if getattr(cfg.MODEL, "DIST_TRAIN", False) and local_rank != 0:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise ImportError(
            "TensorBoard is enabled but not installed. "
            "Install it with `pip install tensorboard`."
        ) from exc

    log_dir = getattr(cfg.TENSORBOARD, "LOG_DIR", "")
    if not log_dir:
        log_dir = os.path.join(output_dir, "tensorboard")
    elif not os.path.isabs(log_dir):
        log_dir = os.path.join(output_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("transreid.train")
    logger.info("TensorBoard logging to: {}".format(log_dir))
    return SummaryWriter(log_dir=log_dir)
