import ee, requests, numpy as np
from matplotlib import pyplot as plt
from pyproj import Transformer
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


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
              scale: int | tuple[int, int],
              crs_epsg: str | None = None,
              patch_size: int = 256,
              add_x_offset: int = 0,
              add_y_offset: int = 0,
              file_format: str = "NUMPY_NDARRAY"):

    # Get patch center in CRS coords
    if crs_epsg is None:
        crs_epsg = get_utm_epsg(pt)
    centroid = wgs84_to_utm(pt, crs_epsg=crs_epsg)

    # scale unpack
    if isinstance(scale, int):
        scale_x, scale_y = scale, -scale
    elif isinstance(scale, tuple) and len(scale) == 2:
        scale_x, scale_y = scale[0], -scale[1]
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
            'crsCode': crs_epsg
        }
    }
    try:
        return ee.data.computePixels(request)
    except Exception as e:
        raise RuntimeError(f"Error fetching patch: {e}")


def get_neighbourhood(image: ee.Image,
                      pt: tuple[float, float],
                      scale: int,
                      crs_epsg: str,
                      patch_size: int = 256,
                      levels: int = 2,
                      file_format: str = "NUMPY_NDARRAY"):

    # Get x,y tuples of coords relative to centroid for choice of level
    offsets = [(x, y) for y in range(-levels, levels + 1) for x in range(-levels, levels + 1)]

    def fetch_patch(offset):
        x, y = offset
        return (x, y), get_patch(image=image, pt=pt, crs_epsg=crs_epsg, scale=scale, patch_size=patch_size,
                                 add_x_offset=x*patch_size, add_y_offset=y*patch_size, file_format=file_format)

    # Get patches at given offset relative to centroid for each x,y tuple
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_patch, offsets)

    return dict(results)


def plot_neighbourhood(patches: dict[tuple[int,int], np.ndarray|dict],
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

        self.platform_configs = {
            "LANDSAT_4": {
                "sr": 'LANDSAT/LT04/C02/T1_L2',
                "toa": 'LANDSAT/LT04/C02/T1_TOA',
                "thermal_band": 'B6',
                "prep_function": self.prep_l47 if self.rescale_bands else self.prep_l47_no_rescale,
            },
            "LANDSAT_5": {
                "sr": 'LANDSAT/LT05/C02/T1_L2',
                "toa": 'LANDSAT/LT05/C02/T1_TOA',
                "thermal_band": 'B6',
                "prep_function": self.prep_l47 if self.rescale_bands else self.prep_l47_no_rescale,
            },
            "LANDSAT_7": {
                "sr": 'LANDSAT/LE07/C02/T1_L2',
                "toa": 'LANDSAT/LE07/C02/T1_TOA',
                "thermal_band": 'B6_VCID_1',
                "prep_function": self.prep_l47 if self.rescale_bands else self.prep_l47_no_rescale,
            },
            "LANDSAT_8": {
                "sr": 'LANDSAT/LC08/C02/T1_L2',
                "toa": 'LANDSAT/LC08/C02/T1_TOA',
                "thermal_band": 'B10',
                "prep_function": self.prep_l89 if self.rescale_bands else self.prep_l89_no_rescale,
            },
            "LANDSAT_9": {
                "sr": 'LANDSAT/LC09/C02/T1_L2',
                "toa": 'LANDSAT/LC09/C02/T1_TOA',
                "thermal_band": 'B10',
                "prep_function": self.prep_l89 if self.rescale_bands else self.prep_l89_no_rescale,
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

    # Prep Landsat 4/5/7 with rescaling
    @staticmethod
    def prep_l47(image: ee.Image) -> ee.Image:
        """Mask clouds, rename bands, and rescale optical bands by EE default (for L4/5/7)."""
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_band = image.select('B6(_VCID_1)?')
        scaled = optical_bands.addBands(thermal_band).select(
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'B6(_VCID_1)?'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
        )
        mask = LandsatSR.get_cloud_mask(image)
        return image.select([]).addBands(scaled).updateMask(mask)

    # Prep Landsat 4/5/7 without rescaling
    @staticmethod
    def prep_l47_no_rescale(image: ee.Image) -> ee.Image:
        """Mask clouds and rename bands without rescaling optical bands (for L4/5/7)."""
        optical_bands = image.select('SR_B.')
        thermal_band = image.select('B6(_VCID_1)?')
        processed = optical_bands.addBands(thermal_band).select(
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'B6(_VCID_1)?'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
        )
        mask = LandsatSR.get_cloud_mask(image)
        return image.select([]).addBands(processed).updateMask(mask)

    # Prep Landsat 8/9 with rescaling
    @staticmethod
    def prep_l89(image: ee.Image) -> ee.Image:
        """Mask clouds, rename bands, and rescale optical bands by EE default (for L8/9)."""
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_band = image.select('B10')
        scaled = optical_bands.addBands(thermal_band).select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'B10'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
        )
        mask = LandsatSR.get_cloud_mask(image)
        return image.select([]).addBands(scaled).updateMask(mask)

    # Processing function for Landsat 8/9 without rescaling
    @staticmethod
    def prep_l89_no_rescale(image: ee.Image) -> ee.Image:
        """Mask clouds and rename bands without rescaling optical bands (for L8/9)."""
        optical_bands = image.select('SR_B.')
        thermal_band = image.select('B10')
        processed = optical_bands.addBands(thermal_band).select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'B10'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
        )
        mask = LandsatSR.get_cloud_mask(image)
        return image.select([]).addBands(processed).updateMask(mask)


class Sentinel2SR:
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 bands: list[str] | None = None,
                 rescale: bool = True,
                 qa_band: str = 'cs_cdf',          # 'cs' or 'cs_cdf'
                 clear_threshold: float = 0.60,    # 0.50–0.65 generally good
                 prefilter_cloud_pct: int | None = 80,
                 keep_qa: bool = False):
        """
        Harmonized SR + Cloud Score+ masking.

        - Collection: COPERNICUS/S2_SR_HARMONIZED (SR scaled by 1e4)
        - Cloud mask: GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED (10 m)
          linked by system:index, using qa_band >= clear_threshold.

        Args:
            bands: Optional band subset (after renaming); e.g. ['RED','GREEN','BLUE'].
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
        s2 = s2.linkCollection(csplus, linkedBands=[qa_band])  # default matchPropertyName='system:index'

        def _prep(img: ee.Image) -> ee.Image:
            # Mask with Cloud Score+ threshold.
            qa = img.select(self.qa_band)
            img = img.updateMask(qa.gte(self.clear_threshold))

            # Select optical bands.
            optical = img.select(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
            if self.rescale:
                optical = optical.multiply(1e-4)

            out = img.select([]).addBands(optical)
            if self.keep_qa:
                out = out.addBands(qa)  # keep 'cs' or 'cs_cdf' for inspection

            return out

        ic = s2.map(_prep)
        if self.bands is not None:
            ic = ic.select(self.bands)

        # Expose both names for convenience (matches your LandsatSR style).
        self.images = ic
        self.collection = ic