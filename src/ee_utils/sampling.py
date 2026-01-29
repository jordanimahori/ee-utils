import re
import ee
import pandas as pd
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from .geo import get_utm_epsg, wgs84_to_utm


def _validate_image_source(image: ee.Image | None, asset_id: str | None):
    if not (image is None) ^ (asset_id is None):
        raise ValueError("Must specify exactly one of image or asset_id")

def _validate_pts(pts, patch_ids):
    if len(pts) != len(patch_ids) or len(patch_ids) != len(set(patch_ids)):
        raise ValueError("pts and patch_ids must have the same length and contain unique elements")
    for pt in pts:
        if not isinstance(pt, tuple) or len(pt) != 2:
            raise ValueError("pts must be a list of tuples (lon, lat)")

def _build_ee_request(
        pt: tuple[float, float],
        pt_crs: str = "EPSG:4326",
        patch_scale: int | tuple[int, int] = 30,
        patch_crs: str | None = None,
        patch_size: int = 256,
        add_x_offset: int = 0,
        add_y_offset: int = 0,
        file_format: str = "NUMPY_NDARRAY"):

    # Map pt to UTM if necessary
    if patch_crs is None:
        if pt_crs.lower() != "epsg:4326":
            raise ValueError("pt_crs must be EPSG:4326 when patch_crs is omitted")
        patch_crs = get_utm_epsg(pt)
        centroid = wgs84_to_utm(pt, crs_epsg=patch_crs)
    elif re.fullmatch(r"(?:epsg:)?(326|327)\d{2}", patch_crs.lower()):
        if pt_crs.lower() == "epsg:4326":
            centroid = wgs84_to_utm(pt, crs_epsg=patch_crs)
        else:
            if pt_crs.lower() != patch_crs.lower():
                raise ValueError("pt_crs must match patch_crs when coordinates are already UTM")
            centroid = pt
    else:
        raise ValueError("patch_crs must be None or a valid UTM EPSG code")

    # Ensure y-axis scale is negative so the grid is north-up
    if isinstance(patch_scale, int):
        scale_x, scale_y = abs(patch_scale), -abs(patch_scale)
    elif isinstance(patch_scale, tuple) and len(patch_scale) == 2:
        scale_x, scale_y = abs(patch_scale[0]), -abs(patch_scale[1])
        if not isinstance(scale_x, int) or not isinstance(scale_y, int):
            raise TypeError("Scale tuple must contain integers")
    else:
        raise TypeError("Scale must be an integer or tuple of two integers")
    assert scale_x > 0 > scale_y

    # Calculate translation
    tx = centroid[0] - scale_x * (patch_size / 2) + scale_x * add_x_offset
    ty = centroid[1] - scale_y * (patch_size / 2) - scale_y * add_y_offset

    request = {
        "fileFormat": file_format,
        "grid": {
            "dimensions": {"width": patch_size, "height": patch_size},
            "affineTransform": {
                "scaleX": scale_x,
                "shearX": 0,
                "translateX": tx,
                "shearY": 0,
                "scaleY": scale_y,
                "translateY": ty,
            },
            "crsCode": patch_crs,
        },
    }
    return request


def get_patch(
    pt: tuple[float, float],
    pt_crs: str = "EPSG:4326",
    image: ee.Image | None = None,
    asset_id: str | None = None,
    patch_scale: int | tuple[int, int] = 30,
    patch_crs: str | None = None,
    patch_size: int = 256,
    add_x_offset: int = 0,
    add_y_offset: int = 0,
    file_format: str = "NUMPY_NDARRAY"
):
    """Fetch a patch centered on pt."""
    request = _build_ee_request(
        pt=pt,
        pt_crs=pt_crs,
        patch_scale=patch_scale,
        patch_crs=patch_crs,
        patch_size=patch_size,
        add_x_offset=add_x_offset,
        add_y_offset=add_y_offset,
        file_format=file_format
    )

    _validate_image_source(image, asset_id)

    if image is not None:           # for computed images, use computePixels
        request["expression"] = image
        try:
            return ee.data.computePixels(request)
        except Exception as exc:
            raise RuntimeError(f"Error fetching patch: {exc}") from exc
    elif asset_id is not None:       # for non-computed images, use getPixels
        request["assetId"] = asset_id
        try:
            return ee.data.getPixels(request)
        except Exception as exc:
            raise RuntimeError(f"Error fetching patch: {exc}") from exc
    else:
        raise ValueError("Must specify either image or asset_id")


def get_neighbourhood(
    pt: tuple[float, float],
    patch_scale: int,
    patch_crs: str,
    image: ee.Image | None = None,
    asset_id: str | None = None,
    pt_crs: str = "EPSG:4326",
    patch_size: int = 256,
    levels: int = 2,
    file_format: str = "NUMPY_NDARRAY",
    concurrent_requests = 20
):
    """Fetch a square neighbourhood of patches around a point."""
    offsets = [(x, y) for y in range(-levels, levels + 1) for x in range(-levels, levels + 1)]

    _validate_image_source(image, asset_id)

    def fetch_patch(offset: tuple[int, int]):
        x, y = offset
        return (
            (x, y),
            get_patch(
                pt=pt,
                pt_crs=pt_crs,
                patch_crs=patch_crs,
                image=image,
                asset_id=asset_id,
                patch_scale=patch_scale,
                patch_size=patch_size,
                add_x_offset=x * patch_size,
                add_y_offset=y * patch_size,
                file_format=file_format,
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        results = executor.map(fetch_patch, offsets)

    return dict(results)


def extract_patches(
    output_dir: str,
    pts: list[tuple[float, float]],
    patch_ids: list[str],
    patch_size: int,
    patch_scale: int,
    image: ee.Image | None = None,
    asset_id: str | None = None,
    patch_crs: str | None = None,
    pts_crs: str = "EPSG:4326",
    overwrite_patches: bool = True,
    concurrent_requests: int = 40,
) -> None:
    """
    Extract patches from Earth Engine and save as a GeoTIFF.

    In case this runs too fast and some requests hit EarthEngine's API rate limits, re-run the export with
    overwrite_patches = False to only download missing patches.
    """

    # Check inputs
    _validate_image_source(image=image, asset_id=asset_id)
    _validate_pts(patch_ids=patch_ids, pts=pts)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not overwrite_patches:
        existing_names = {p.stem for p in Path(output_dir).glob("*.tif")}
        skip_ids = [name for name in patch_ids if name in existing_names]
        if len(skip_ids) == len(patch_ids):
            print("...all patches already exist.")
            return
        if skip_ids:
            print(f"...skipping {len(skip_ids)} existing patches.")
    else:
        skip_ids = []

    future_to_point: dict[concurrent.futures.Future, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        for pt, pt_id in zip(pts, patch_ids):
            if pt_id not in skip_ids:
                future = executor.submit(
                    get_patch,
                    image=image,
                    asset_id=asset_id,
                    pt=pt,
                    pt_crs=pts_crs,
                    patch_scale=patch_scale,
                    patch_size=patch_size,
                    patch_crs=patch_crs,
                    file_format="GEO_TIFF",
                )
                future_to_point[future] = pt_id

        progbar = tqdm(total=len(future_to_point))
        for future in concurrent.futures.as_completed(future_to_point):
            pt_id = future_to_point[future]

            try:
                img = future.result()
                filepath = Path(output_dir).joinpath(f"{pt_id}.tif")
                with open(filepath, "wb") as writer:
                    writer.write(img)

            except Exception as exc:  # noqa: BLE001
                print(exc)

            finally:
                future_to_point.pop(future)
                progbar.update(1)

        progbar.close()


def extract_spatial_covariates(
    pts: list[tuple[float, float]],
    patch_ids: list[str],
    patch_scale: int,
    image: ee.Image | None = None,
    asset_id: str | None = None,
    patch_crs: str | None = None,
    pt_crs: str = "EPSG:4326",
    concurrent_requests: int = 40,
) -> pd.DataFrame:
    """
    Extract average covariate values for a list of points and save as a Pandas DataFrame.

    In case this runs too fast and some requests hit EarthEngine's API rate limits, re-run the export with
    overwrite_patches = False to only download missing patches.
    """

    # Check inputs
    _validate_image_source(image=image, asset_id=asset_id)
    _validate_pts(patch_ids=patch_ids, pts=pts)

    future_to_point: dict[concurrent.futures.Future, str] = {}
    patch_dfs: list[pd.DataFrame] = []
    errors: list[tuple[str, Exception]] = []
    with tqdm(total=len(pts)) as progbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            for pt, pt_id in zip(pts, patch_ids):
                future = executor.submit(
                    get_patch,
                    image=image,
                    asset_id=asset_id,
                    pt=pt,
                    pt_crs=pt_crs,
                    patch_scale=patch_scale,
                    patch_size=1,
                    patch_crs=patch_crs,
                )
                future_to_point[future] = pt_id

            for future in concurrent.futures.as_completed(future_to_point):
                pt_id = future_to_point[future]
                try:
                    img = future.result()
                    df = pd.DataFrame.from_records(img.reshape(-1))
                    df["patch_id"] = pt_id
                    patch_dfs.append(df)

                except Exception as exc:  # noqa: BLE001
                    errors.append((pt_id, exc))

                finally:
                    future_to_point.pop(future)
                    progbar.update()

    if errors:
        raise RuntimeError(
            f"{len(errors)} patches failed. First error: {errors[0][0]} → {errors[0][1]!r}"
        )
    return pd.concat(patch_dfs)
