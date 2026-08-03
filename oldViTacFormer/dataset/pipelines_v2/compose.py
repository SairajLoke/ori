import importlib
import collections

def instantiate_from_config(config):
    """
    Instantiate a class or function from config.
    Supports both:
    - {"target": "xxx.Class", "params": {...}}
    - {"target": "xxx.Class", "arg1": ..., "arg2": ...}
    """
    assert "target" in config, "Config must have a 'target' field"
    target = config["target"]

    # Try both param styles
    params = dict(config)  # shallow copy
    params.pop("target")
    params.pop("params", None)  # prevent conflict

    # If "params" is used, update with that
    if "params" in config:
        params.update(config["params"])

    module_path, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls_or_func = getattr(module, cls_name)

    return cls_or_func(**params)


class Compose:
    """
    Compose multiple transforms sequentially.

    Supports:
    - Callable transforms
    - Config dicts with "target" and optional "params"
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence), type(transforms)
        self.transforms = []
        for transform in transforms:
            # print(transform)
            if isinstance(transform, dict) and "target" in transform:
                transform = instantiate_from_config(transform)
            elif not callable(transform):
                raise TypeError(f"Transform must be callable or a config dict with 'target'. Got {type(transform)}")
            self.transforms.append(transform)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
