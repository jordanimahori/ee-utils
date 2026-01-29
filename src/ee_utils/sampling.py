import concurrent.futures
import re
from pathlib import Path

import ee
import pandas as pd
from tqdm import tqdm
from .geo import get_utm_epsg, wgs84_to_utm


def get_patch(image: ee.Image,
              pt: tuple[float, float],
              pt_crs: str = "EPSG:4326",
              patch_scale: int | tuple[int, int] = 30,
              patch_crs: str | None = None,
              patch_size: int = 256,
              add_x_offset: int = 0,
              add_y_offset: int = 0,
              file_format: str = "NUMPY_NDARRAY"):

    # Get patch center in UTM coordinates for zone corresponding to supplied pt
    if patch_crs is None:
        assert pt_crs.lower() == "epsg:4326"
        patch_crs = get_utm_epsg(pt)
        centroid = wgs84_to_utm(pt, crs_epsg=patch_crs)
    elif re.fullmatch(r"(?:epsg:)?(326|327)\d{2}", patch_crs.lower()):
        if pt_crs.lower() == "epsg:4326":                        # case 1: convert WGS84 to corresponding UTM
            centroid = wgs84_to_utm(pt, crs_epsg=patch_crs)
        else:                                                    # case 2: pt already in correct UTM
            assert pt_crs.lower() == patch_crs.lower()
            centroid = pt
    else:
        raise NotImplementedError

    # patch_scale unpack
    if isinstance(patch_scale, int):
        scale_x, scale_y = patch_scale, -patch_scale
    elif isinstance(patch_scale, tuple) and len(patch_scale) == 2:
        scale_x, scale_y = patch_scale[0], -patch_scale[1]
        assert isinstance(scale_x, int) and isinstance(scale_y, int)
    else:
        raise TypeError("Scale must be an integer or tuple of two integers")
    assert scale_x > 0 > scale_y

    # Get translation for patch centroid
    tx = centroid[0] - scale_x * (patch_size/2) + scale_x * add_x_offset
    ty = centroid[1] - scale_y * (patch_size/2) + -scale_y * add_y_offset

    request = {
        'expression': image,
        'fileFormat': file_format,
        'grid': {
            'dimensions': {'width': patch_size, 'height': patch_size},
            'affineTransform': {
                'scaleX': scale_x, 'shearX': 0, 'translateX': tx,
                'shearY': 0,      'scaleY': scale_y, 'translateY': ty
            },
            'crsCode': patch_crs
        }
    }
    try:
        return ee.data.computePixels(request)
    except Exception as e:
        raise RuntimeError(f"Error fetching patch: {e}")


def get_neighbourhood(image: ee.Image,
                      pt: tuple[float, float],
                      patch_scale: int,
                      patch_crs: str,
                      pt_crs: str = "EPSG:4326",
                      patch_size: int = 256,
                      levels: int = 2,
                      file_format: str = "NUMPY_NDARRAY"):

    # Get x,y tuples of coordinates relative to centroid given choice of level
    offsets = [(x, y) for y in range(-levels, levels + 1) for x in range(-levels, levels + 1)]

    def fetch_patch(offset):
        x, y = offset
        return (x, y), get_patch(image=image, pt=pt, pt_crs=pt_crs, patch_crs=patch_crs, patch_scale=patch_scale, patch_size=patch_size,
                                 add_x_offset=x*patch_size, add_y_offset=y*patch_size, file_format=file_format)

    # Get patches at given offset relative to centroid for each x,y tuple
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fetch_patch, offsets)

    return dict(results)


def extract_patches(image: ee.Image,
                    output_dir: str,
                    pts: list[tuple[float, float]],
                    patch_names: list[str],
                    patch_size: int,
                    patch_scale: int,
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
    assert len(pts) == len(patch_names)
    assert len(patch_names) == len(set(patch_names))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Skip existing patches if names match
    if not overwrite_patches:
        existing_names = {p.stem for p in Path(output_dir).glob("*.tif")}
        skip_ids = [n for n in patch_names if n in existing_names]
        # and return early if all IDs to be skipped
        if len(skip_ids) == len(patch_names):
            print("...all patches already exist.")
            return None
        if len(skip_ids) > 0:
            print(f"...skipping {len(skip_ids)} existing patches.")
    else:
        skip_ids = []

    future_to_point = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        for pt, pt_id in zip(pts, patch_names):
            if pt_id not in skip_ids:
                future = executor.submit(get_patch, image=image, pt=pt, pt_crs=pts_crs, patch_scale=patch_scale,
                                         patch_size=patch_size, patch_crs=patch_crs, file_format='GEO_TIFF')
                future_to_point[future] = pt_id

        # Write patches to disk when available
        progbar = tqdm(total=len(future_to_point))
        for future in concurrent.futures.as_completed(future_to_point):
            pt_id = future_to_point[future]

            try:
                img = future.result()
                filepath = Path(output_dir).joinpath(f"{pt_id}.tif")
                with open(filepath, 'wb') as writer:
                    writer.write(img)

            except Exception as e:
                print(e)

            finally:
                future_to_point.pop(future)
                progbar.update(1)

        progbar.close()


def extract_spatial_covariates(image: ee.Image,
                               pts: list[tuple[float, float]],
                               patch_ids: list[str],
                               patch_scale: int,
                               patch_crs: str | None = None,
                               pt_crs: str = "EPSG:4326",
                               concurrent_requests: int = 40
                               ) -> pd.DataFrame:
    """
    Extract average covariate values for a list of points and save as a Pandas DataFrame.

    In case this runs too fast and some requests hit EarthEngine's API rate limits, re-run the export with
    overwrite_patches = False to only download missing patches.
    """

    assert len(pts) == len(patch_ids)

    future_to_point = {}
    patch_dfs = []
    errors = []
    with tqdm(total=len(pts)) as progbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:

            # Submit jobs
            for pt, pt_id in zip(pts, patch_ids):
                    future = executor.submit(get_patch, image=image, pt=pt, pt_crs=pt_crs, patch_scale=patch_scale,
                                             patch_size=1, patch_crs=patch_crs)
                    future_to_point[future] = pt_id

            # Write patches to disk when available
            for future in concurrent.futures.as_completed(future_to_point):
                pt_id = future_to_point[future]
                try:
                    img = future.result()
                    df =  pd.DataFrame.from_records(img.reshape(-1))
                    df["patch_id"] = pt_id
                    patch_dfs.append(df)

                except Exception as e:
                    errors.append((pt_id, e))

                finally:
                    future_to_point.pop(future)
                    progbar.update()

    if errors:
        raise RuntimeError(
            f"{len(errors)} patches failed. "
            f"First error: {errors[0][0]} → {repr(errors[0][1])}"
        )
    return pd.concat(patch_dfs)
