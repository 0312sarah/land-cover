from .unet import UNetTFStyle

def build_model(model_cfg: dict):
    name = model_cfg["name"].lower()

    if name == "unet":
        return UNetTFStyle(
            in_channels=model_cfg.get("in_channels", 4),
            num_classes=model_cfg.get("num_classes", 10),
            num_layers=model_cfg.get("num_layers", 4),
            base_channels=model_cfg.get("base_channels", 64),
            upconv_filters=model_cfg.get("upconv_filters", 96),
        )

    raise ValueError(f"Unknown model name: {model_cfg['name']}")