import logging

try:
    from yacs.config import CfgNode as CN
except Exception:
    CN = None


def _cfg_to_dict(cfg):
    if CN is not None and isinstance(cfg, CN):
        return {k: _cfg_to_dict(v) for k, v in cfg.items()}
    if hasattr(cfg, "items") and not isinstance(cfg, (str, bytes)):
        try:
            return {k: _cfg_to_dict(v) for k, v in cfg.items()}
        except Exception:
            pass
    return cfg


def init_wandb(cfg, output_dir, run_name, local_rank=0):
    if not hasattr(cfg, "WANDB"):
        return None
    if not cfg.WANDB.ENABLED:
        return None
    if getattr(cfg.MODEL, "DIST_TRAIN", False) and local_rank != 0:
        return None

    try:
        import wandb
    except ImportError:
        logging.getLogger("transreid.train").warning(
            "wandb is not installed; skipping wandb logging."
        )
        return None

    tags = cfg.WANDB.TAGS
    if tags is None:
        tags_list = None
    elif isinstance(tags, (list, tuple)):
        tags_list = list(tags)
    else:
        tags_list = [str(tags)]

    return wandb.init(
        project=cfg.WANDB.PROJECT,
        entity=cfg.WANDB.ENTITY or None,
        name=run_name or None,
        dir=output_dir or None,
        tags=tags_list,
        config=_cfg_to_dict(cfg),
    )
