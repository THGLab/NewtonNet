import os
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


def download_checkpoint(model: str):
    """
    Downloads or locates the NewtonNet checkpoint file.
    Based on https://github.com/ACEsuit/mace/blob/main/mace/calculators/foundations_models.py

    Args:
        model (str): Path to the model specification.

    Returns:
        str: Path to the downloaded (or cached, if previously loaded) checkpoint file.
    """
    urls = {
        "ani1": "https://github.com/THGLab/NewtonNet/releases/download/pretrained/newtonnet_ani1.zip",
        "ani1x": "https://github.com/THGLab/NewtonNet/releases/download/pretrained/newtonnet_ani1x.zip",
        "t1x": "https://github.com/THGLab/NewtonNet/releases/download/pretrained/newtonnet_t1x.zip",
    }

    checkpoint_url = urls.get(model, model)

    cache_dir = os.path.expanduser("~/.cache/newtonnet")
    cached_zip_path = os.path.join(cache_dir, f"newtonnet_{model}.zip")
    cached_model_path = os.path.join(cache_dir, f"newtonnet_{model}/models/best_model.pt")

    if not os.path.exists(cached_model_path):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Downloading NewtonNet model from {checkpoint_url!r}")
        _, http_msg = urlretrieve(checkpoint_url, cached_zip_path)
        if "Content-Type: text/html" in http_msg:
            raise RuntimeError(
                f"Model download failed, please check the URL {checkpoint_url}"
            )
        with zipfile.ZipFile(cached_zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
        os.remove(cached_zip_path)
        print(f"Cached NewtonNet model to {cached_model_path}")

    return cached_model_path
