import ee, requests, re, numpy as np, pandas as pd
from matplotlib import pyplot as plt
from pyproj import Transformer
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import concurrent.futures


def get_utm_epsg(pt):
    """Determine the UTM EPSG code for a given lon, lat pair in WGS84."""
    lon, lat = pt
    assert (-180 <= lon <= 180 and -80 <= lat <= 84),  "UTM only defined for latitudes between -80 and 84 degrees"
    lon = 179.999999 if lon == 180 else lon                            # avoid zone=61 at the antimeridian
    utm_zone = int((lon + 180) // 6) + 1                               # UTM zone number (ignoring Svalbaard)
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone     # 326xx for NH and 327xx for SH
    return f"EPSG:{epsg_code}"


def wgs84_to_utm(pt: tuple[float, float], crs_epsg: str) -> tuple[float, float]:
    """“Convert lon, lat (WGS84) to UTM coordinates in the requested zone."""
    transformer = Transformer.from_crs(crs_from="epsg:4326", crs_to=crs_epsg, always_xy=True)
    return transformer.transform(*pt)


def get_patch(image: ee.Image,
              pt: tuple[float, float],
              pt_crs: str = "EPSG:4326",
              patch_scale: int | tuple[int, int] = 30,
              patch_crs: str | None = None,
              patch_size: int = 256,
              add_x_offset: int = 0,
              add_y_offset: int = 0,
              file_format: str = "NUMPY_NDARRAY"):

    # Get patch center in UTM coords; assume WGS84 if none supplied
    if patch_crs is None:
        assert pt_crs.lower() == "epsg:4326"
        patch_crs = get_utm_epsg(pt)
        centroid = wgs84_to_utm(pt, crs_epsg=patch_crs)
    elif re.fullmatch(r"(?:epsg:)?(326|327)\d{2}", patch_crs.lower()):
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

    # Get translation for patch center
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

    # Get x,y tuples of coords relative to centroid for choice of level
    offsets = [(x, y) for y in range(-levels, levels + 1) for x in range(-levels, levels + 1)]

    def fetch_patch(offset):
        x, y = offset
        return (x, y), get_patch(image=image, pt=pt, pt_crs=pt_crs, patch_crs=patch_crs, patch_scale=patch_scale, patch_size=patch_size,
                                 add_x_offset=x*patch_size, add_y_offset=y*patch_size, file_format=file_format)

    # Get patches at given offset relative to centroid for each x,y tuple
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_patch, offsets)

    return dict(results)


def plot_neighbourhood(patches: dict[tuple[int,int], np.ndarray],
                       levels: int,
                       bands: str | list[str],
                       vis_min: float,
                       vis_max: float,
                       **kwargs
                       ) -> None:
    if isinstance(bands, str):
        bands = [bands]
    assert len(bands) in [1, 3], "Visualisation only possible for 1 or 3 bands"
    if levels == 0 and isinstance(patches, np.ndarray):
        patches = {(0, 0): patches}

    _, axs = plt.subplots(2*levels + 1, 2*levels + 1, figsize=(12, 12))
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
                raise ValueError('supplied array is of an unsupported type')

            # Normalize
            display_array = (np.clip(display_array, vis_min, vis_max) - vis_min)/(vis_max - vis_min)

            if len(bands) == 1:
                display_array = display_array[..., 0]

            # Assign to i,j subplot
            if levels == 0:
                axs.imshow(display_array, vmin=0, vmax=1, **kwargs)
                axs.set_xticks([])
                axs.set_yticks([])

            else:
                axs[-y + levels, x + levels].imshow(display_array, vmin=0, vmax=1, **kwargs)
                axs[-y + levels, x + levels].set_xticks([])
                axs[-y + levels, x + levels].set_yticks([])

    plt.show()


def preview_patch(image: ee.Image,
                  pt: tuple[float, float],
                  preset: str,
                  scale: int,
                  patch_size=256):
    """PNG preview centered at pt."""
    param_dict = {
        'sentinel1': [{'bands': ['VV'], 'min': -15, 'max': 0}, 'gray'],
        'sentinel2': [{'bands': ['B4','B3','B2'],'min': 0.0, 'max': 0.3}, 'rgb'],
        'landsat': [{'bands': ['BLUE','GREEN','RED'], 'min': 0, 'max': 0.4}, 'rgb'],
    }
    preset = preset.lower()
    assert preset in param_dict, f"Unknown preset: {preset}"
    vis_params, _ = param_dict[preset]

    pt_geom = ee.Geometry.Point(pt)
    region = pt_geom.buffer((patch_size/2)*scale).bounds()

    url = image.getThumbURL({
        'region': region,
        'dimensions': f'{patch_size}x{patch_size}',
        'format': 'png',
        'min': vis_params['min'],
        'max': vis_params['max'],
        'bands': vis_params['bands']
    })

    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError("Failed to retrieve image.")
    img = Image.open(BytesIO(r.content))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


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
                               concurrent_requests: int = 40,
                               **kwargs
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:

        # Submit jobs
        for pt, pt_id in zip(pts, patch_ids):
                future = executor.submit(get_patch, image=image, pt=pt, pt_crs=pt_crs, patch_scale=patch_scale,
                                         patch_size=1, patch_crs=patch_crs, **kwargs)
                future_to_point[future] = pt_id

        # Write patches to disk when available
        progbar = tqdm(total=len(future_to_point))
        for future in concurrent.futures.as_completed(future_to_point):
            pt_id = future_to_point[future]

            try:
                img = future.result()
                df = pd.DataFrame(img.reshape(-1))
                df["dhs_id"] = pt_id
                patch_dfs.append(df)

            except Exception as e:
                print(f"Encountered error for {pt_id}:")
                errors.append(pt_id)
                print(e)

            finally:
                future_to_point.pop(future)
                progbar.update(1)

        progbar.close()

    return pd.concat(patch_dfs), errors



def prefix_bands(img, prefix):
    new_names = img.bandNames().map(lambda b: ee.String(prefix).cat(ee.String(b)))
    return img.rename(new_names)


class SpatialCovariates:

    # Earliest and latest available data for each ImageCollection
    TERRACLIMATE_START, TERRACLIMATE_END = 1958, 2024  # for "IDAHO_EPSCOR/TERRACLIMATE"
    GHS_START, GHS_END = 1975, 2030  # for "JRC/GHSL/P2023A/GHS_POP"; 'JRC/GHSL/P2023A/GHS_BUILT_S' -- (only every five years)
    VIIRS_START, VIIRS_END = 2012, 2024  # for "NASA/VIIRS/002/VNP46A2"
    CHIRPS_START, CHIRPS_END = 1981, 2025  # for "UCSB-CHG/CHIRPS/PENTAD"
    DYNWORLD_START, DYNWORLD_END = 2016, 2025  # for "GOOGLE/DYNAMICWORLD/V1"; 2016 is the first full year of data

    def __init__(self, year: int,
                 landcover_scale: int = 100,
                 nodata_val: int = -9999,
                 max_pixels: int = 64,
                 best_effort: bool = True):
        self.year = year
        self.cov_years = {
            "terraclimate": int(np.clip(self.year, self.TERRACLIMATE_START, self.TERRACLIMATE_END)),
            "ghs": int(np.clip((self.year // 5) * 5, self.GHS_START, self.GHS_END)),        # GHS only available every five years
            "viirs": int(np.clip(self.year, self.VIIRS_START, self.VIIRS_END)),
            "chirps": int(np.clip(self.year, self.CHIRPS_START, self.CHIRPS_END)),
            "dynworld": int(np.clip(self.year, self.DYNWORLD_START, self.DYNWORLD_END))
        }
        self.landcover_scale = landcover_scale
        self.nodata_val = nodata_val
        self.max_pixels = max_pixels
        self.best_effort = best_effort

        # --- Load ImageCollections ---
        terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(f"{self.cov_years['terraclimate']}",f"{self.cov_years['terraclimate'] + 1}").select(["tmmx", "tmmn"])
        ghs_built_surfaces = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S").filterDate(f"{self.cov_years['ghs']}",f"{self.cov_years['ghs'] + 1}")
        ghs_population = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").filterDate(f"{self.cov_years['ghs']}",f"{self.cov_years['ghs'] + 1}")
        viirs_nightlights = ee.ImageCollection("NASA/VIIRS/002/VNP46A2").filterDate(f"{self.cov_years['viirs']}",f"{self.cov_years['viirs'] + 1}").select(["DNB_BRDF_Corrected_NTL"], ["viirs"])
        precipitation = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(f"{self.cov_years['chirps']}",f"{self.cov_years['chirps'] + 1}")
        land_cover = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(f"{self.cov_years['dynworld']}",f"{self.cov_years['dynworld'] + 1}")

        # --- Create Annual Composite Images ---
        # Note: We use ee.Image.reduceResolution to control downsampling; sum for extensive and mean for intensive quantities.
        #       We need a reference proj for .reduceResolution so we know which of the pixels from our temporal composite image
        #       (which may have different projections) fall within each cell in the requested grid. For global compatibility,
        #       we use EPSG:4326 if no IC-wide proj is available, but note that this has some implications for weighting of the final composite.

        # Reference Proj for Composites
        tc_default = terraclimate.first().projection()
        viirs_default = viirs_nightlights.first().projection()
        precip_default = precipitation.first().projection()
        lc_default = ee.Projection('EPSG:4326').atScale(self.landcover_scale)    # WGS84; DW otherwise has varying UTM projs

        # Reduce using mean, max, min, and stdDev
        reducer_multi = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True).combine(ee.Reducer.min(), sharedInputs=True).combine(ee.Reducer.stdDev(), sharedInputs=True)

        # Precipitation
        self.precip_comp = (precipitation
                            .reduce(reducer=reducer_multi)
                            .setDefaultProjection(precip_default)
                            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=self.max_pixels, bestEffort=self.best_effort)
                            .unmask(self.nodata_val))

        # Land Cover (Dynamic World)
        landcover_prob_comp = (land_cover
                               .select(['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'])
                               .mean()
                               .setDefaultProjection(lc_default)
                               .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=self.max_pixels, bestEffort=self.best_effort)
                               .unmask(self.nodata_val))
        self.landcover_comp = prefix_bands(landcover_prob_comp, "lulc_prob_")

        # Nightlights (VIIRS)
        self.nightlights_comp = (viirs_nightlights
                                 .reduce(reducer=reducer_multi)
                                 .setDefaultProjection(viirs_default)
                                 .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=self.max_pixels, bestEffort=self.best_effort)
                                 .unmask(self.nodata_val))

        # Global Human Settlements
        pop_comp = ghs_population.first().reduceResolution(reducer=ee.Reducer.sum(), maxPixels=self.max_pixels, bestEffort=self.best_effort).unmask(self.nodata_val)
        settlement_comp = ghs_built_surfaces.first().reduceResolution(reducer=ee.Reducer.sum(), maxPixels=self.max_pixels, bestEffort=self.best_effort).unmask(self.nodata_val)
        self.ghs_comp = ee.Image.cat([pop_comp, settlement_comp])

        # Terraclimate
        self.terraclimate_comp = (terraclimate
                                  .reduce(reducer=reducer_multi)
                                  .setDefaultProjection(tc_default)
                                  .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=self.max_pixels, bestEffort=self.best_effort)
                                  .unmask(self.nodata_val))

        # Soil PH (static image)
        self.soil_ph_comp = (ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
                             .select("b0")
                             .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=self.max_pixels, bestEffort=self.best_effort)
                             .unmask(self.nodata_val))

        # Store Composite Images
        self.all_covariates_comp = ee.Image.cat([self.landcover_comp, self.nightlights_comp, self.ghs_comp,
                                                 self.terraclimate_comp, self.precip_comp, self.soil_ph_comp]).toFloat()


class LandsatSR:
    def __init__(self, start_date: str,
                 end_date: str,
                 bands: list[str] | None = None,
                 platforms: list[str] | None = None,
                 rescale_bands: bool = True
                 ) -> None:
        """
        Args:
            start_date: String representation of start date.
            end_date: String representation of end date.
            bands (Optional): List of bands to select.
            platforms (Optional): List of Landsat platforms (e.g. ["LANDSAT_8", "LANDSAT_9"]).
            rescale_bands: If true, rescales Landsat bands by USGS recommended values.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.bands = bands
        self.rescale_bands = rescale_bands
        self.available_platforms = {"LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8", "LANDSAT_9"}
        self.available_bands = {'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1'}

        # If no platforms provided, use all available
        if platforms is None:
            self.platforms = self.available_platforms
        elif isinstance(platforms, str):
            self.platforms = {platforms.upper()}
        else:
            self.platforms = {p.upper() for p in platforms}

        # Check arguments
        if not self.platforms.issubset(self.available_platforms):
            raise ValueError(f"Platforms must be a subset of {self.available_platforms}")

        # Prep Landsat 4/5/7 with rescaling
        def _prep_l47(image: ee.Image) -> ee.Image:
            """Mask clouds, rename bands, and rescale optical bands by EE default (for L4/5/7)."""
            optical_bands = image.select('SR_B.')
            if self.rescale_bands:
                optical_bands = optical_bands.multiply(0.0000275).add(-0.2)
            thermal_band = image.select('B6(_VCID_1)?')
            scaled = optical_bands.addBands(thermal_band).select(
                ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'B6(_VCID_1)?'],
                ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
            )
            mask = LandsatSR.get_cloud_mask(image)
            return image.select([]).addBands(scaled).updateMask(mask)

        # Prep Landsat 8/9 with rescaling
        def _prep_l89(image: ee.Image) -> ee.Image:
            """Mask clouds, rename bands, and rescale optical bands by EE default (for L8/9)."""
            optical_bands = image.select('SR_B.')
            if self.rescale_bands:
                optical_bands = optical_bands.multiply(0.0000275).add(-0.2)
            thermal_band = image.select('B10')
            scaled = optical_bands.addBands(thermal_band).select(
                ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'B10'],
                ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
            )
            mask = LandsatSR.get_cloud_mask(image)
            return image.select([]).addBands(scaled).updateMask(mask)


        self.platform_configs = {
            "LANDSAT_4": {
                "sr": 'LANDSAT/LT04/C02/T1_L2',
                "toa": 'LANDSAT/LT04/C02/T1_TOA',
                "thermal_band": 'B6',
                "prep_function": _prep_l47
            },
            "LANDSAT_5": {
                "sr": 'LANDSAT/LT05/C02/T1_L2',
                "toa": 'LANDSAT/LT05/C02/T1_TOA',
                "thermal_band": 'B6',
                "prep_function": _prep_l47
            },
            "LANDSAT_7": {
                "sr": 'LANDSAT/LE07/C02/T1_L2',
                "toa": 'LANDSAT/LE07/C02/T1_TOA',
                "thermal_band": 'B6_VCID_1',
                "prep_function": _prep_l47
            },
            "LANDSAT_8": {
                "sr": 'LANDSAT/LC08/C02/T1_L2',
                "toa": 'LANDSAT/LC08/C02/T1_TOA',
                "thermal_band": 'B10',
                "prep_function": _prep_l89
            },
            "LANDSAT_9": {
                "sr": 'LANDSAT/LC09/C02/T1_L2',
                "toa": 'LANDSAT/LC09/C02/T1_TOA',
                "thermal_band": 'B10',
                "prep_function": _prep_l89
            },
        }

        # Prep and merge ImageCollections for requested platforms
        self.images = ee.ImageCollection([])
        for platform in self.platforms:
            config = self.platform_configs[platform]
            col = (
                ee.ImageCollection(config["sr"])
                .filterDate(self.start_date, self.end_date)
                .linkCollection(                               # join thermal band from TOA ee.ImageCollection
                    ee.ImageCollection(config["toa"]),
                    linkedBands=[config["thermal_band"]]
                )
                .map(config["prep_function"])                  # mask low-quality pixels (and optionally rescale)
            )
            # If bands are specified, filter the ee.ImageCollection.
            if self.bands is not None:
                col = col.select(self.bands)
            self.images = self.images.merge(col)

    @staticmethod
    def get_cloud_mask(image: ee.Image) -> ee.Image:
        """Get mask for non-cloudy pixels for a Landsat image based on QA_PIXEL."""
        qa = image.select(['QA_PIXEL'])
        cloud_bit_mask = (1 << 3)
        cloud_shadow_bit_mask = (1 << 4)
        return qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
        )


class Sentinel2SR:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 bands: list[str] | None = None,
                 rescale: bool = True,
                 qa_band: str = 'cs_cdf',
                 clear_threshold: float = 0.60,
                 prefilter_cloud_pct: int | None = 80,
                 keep_qa: bool = False):
        """
        Harmonized SR + Cloud Score+ masking.

        - Collection: COPERNICUS/S2_SR_HARMONIZED (SR scaled by 1e4)
        - Cloud mask: GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED (10 m)
          linked by system:index, using qa_band >= clear_threshold.

        Args:
            bands: Optional band subset (e.g. ['B2', 'B3', 'B4', 'B8']).
            rescale: Multiply reflectance by 1e-4 to get [0..1] floats.
            qa_band: 'cs' or 'cs_cdf' (recommended: 'cs_cdf').
            clear_threshold: keep pixels with qa_band >= this value.
            prefilter_cloud_pct: optional pre-filter on CLOUDY_PIXEL_PERCENTAGE.
            keep_qa: if True, retain the QA band in outputs for debugging.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.bands = bands
        self.rescale = rescale
        self.qa_band = qa_band
        self.clear_threshold = ee.Number(clear_threshold)
        self.keep_qa = keep_qa

        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(start_date, end_date))
        if prefilter_cloud_pct is not None:
            s2 = s2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', prefilter_cloud_pct))

        # Cloud Score+ (10 m), shares system:index; link the chosen QA band.
        csplus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        s2 = s2.linkCollection(csplus, linkedBands=[qa_band])

        def _prep(img: ee.Image) -> ee.Image:
            # Mask with Cloud Score+ threshold.
            qa = img.select(self.qa_band)
            img = img.updateMask(qa.gte(self.clear_threshold))

            # Select optical bands.
            optical = img.select(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
            if self.rescale:
                optical = optical.toFloat().multiply(ee.Number(1e-4).float())

            out = img.select([]).addBands(optical)
            if self.keep_qa:
                out = out.addBands(qa)  # keep 'cs' or 'cs_cdf' for inspection
            return out

        ic = s2.map(_prep)
        if self.bands is not None:
            ic = ic.select(list(self.bands) + [self.qa_band]) if self.keep_qa else ic.select(self.bands)

        self.images = ic
        self.collection = ic