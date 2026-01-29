from io import BytesIO

import ee
import numpy as np
import requests
from matplotlib import pyplot as plt
from PIL import Image


def preview_patch(
    image: ee.Image,
    pt: tuple[float, float],
    preset: str,
    scale: int,
    patch_size: int = 256,
) -> None:
    """PNG preview centered at pt."""
    param_dict = {
        "sentinel1": [{"bands": ["VV"], "min": -15, "max": 0}, "gray"],
        "sentinel2": [{"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3}, "rgb"],
        "landsat": [{"bands": ["BLUE", "GREEN", "RED"], "min": 0, "max": 0.4}, "rgb"],
    }
    preset = preset.lower()
    if preset not in param_dict:
        raise ValueError(f"Unknown preset: {preset}")
    vis_params, _ = param_dict[preset]

    pt_geom = ee.Geometry.Point(pt)
    region = pt_geom.buffer((patch_size / 2) * scale).bounds()

    url = image.getThumbURL(
        {
            "region": region,
            "dimensions": f"{patch_size}x{patch_size}",
            "format": "png",
            "min": vis_params["min"],
            "max": vis_params["max"],
            "bands": vis_params["bands"],
        }
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to retrieve image.")
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def plot_neighbourhood(
    patches: dict[tuple[int, int], np.ndarray],
    levels: int,
    bands: str | list[str],
    vis_min: float,
    vis_max: float,
    **kwargs,
) -> None:
    if isinstance(bands, str):
        bands = [bands]
    if len(bands) not in {1, 3}:
        raise ValueError("Visualisation only possible for 1 or 3 bands")
    if levels == 0 and isinstance(patches, np.ndarray):
        patches = {(0, 0): patches}

    _, axs = plt.subplots(2 * levels + 1, 2 * levels + 1, figsize=(12, 12))
    plt.subplots_adjust(wspace=0, hspace=0)

    # For each row in the grid defined by levels (start in top-left corner)
    for y in range(levels, -levels - 1, -1):
        # and each column
        for x in range(-levels, levels + 1):
            # Get x,y patch
            patch = patches[(x, y)]

            # Extract desired bands and structure as Numpy Array
            if isinstance(patch, np.ndarray):
                display_array = np.stack([patch[band] for band in bands], 2)
            elif isinstance(patch, dict):
                display_array = np.stack([patch[band].numpy() for band in bands], 2)
            else:
                raise ValueError("supplied array is of an unsupported type")

            display_array = (np.clip(display_array, vis_min, vis_max) - vis_min) / (
                vis_max - vis_min
            )

            if len(bands) == 1:
                display_array = display_array[..., 0]

            # Assign to i,j subplot
            if levels == 0:
                axs.imshow(display_array, vmin=0, vmax=1, **kwargs)
                axs.set_xticks([])
                axs.set_yticks([])

            else:
                axs[-y + levels, x + levels].imshow(
                    display_array, vmin=0, vmax=1, **kwargs
                )
                axs[-y + levels, x + levels].set_xticks([])
                axs[-y + levels, x + levels].set_yticks([])

    plt.show()
