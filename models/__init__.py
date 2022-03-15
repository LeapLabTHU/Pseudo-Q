from .trans_vg_mlcma import TransVG_MLCMA


def build_model(args):
    return TransVG_MLCMA(args)
