import ee
import numpy as np
import requests
from matplotlib import pyplot as plt
from pyproj import Transformer
from typing import Union, Tuple, Optional
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def get_utm_epsg(pt: Tuple[float, float]) -> Optional[str]:
    """
    Determine the UTM EPSG code for a given longitude and latitude.
    """
    lon, lat = pt
    assert (-180 <= lon <= 180 and -80 <= lat <= 84),  "UTM defined for latitudes between -80 and 84 degrees"

    utm_zone = int((lon + 180) / 6) + 1                             # UTM zone number (ignoring Svalbaard)
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone  # 326xx for SH and 327xx for NH

    return f"EPSG:{epsg_code}"


def wgs84_to_utm(pt: tuple[float, float], crs_epsg: str) -> tuple[float, float]:
    """
    Convert latitude and longitude to UTM coordinates.
    """
    # Convert coordinates from WGS84 to UTM
    transformer = Transformer.from_crs(crs_from="epsg:4326", crs_to=crs_epsg, always_xy=True)
    return transformer.transform(*pt)


def preview_patch(image: ee.Image,
                  pt: tuple[float, float],
                  preset: str,
                  scale: int = 10,
                  patch_size=256
                  ):
    """
    Gets URL for an image at designated point, with defaults for Landsat images.
    """

    # Get Vis Params
    param_dict = {
        'sentinel1': [{'bands': ['VV'], 'min': -15, 'max': 0,}, "grey"],
    }
    assert preset.lower() in ["sentinel1"]
    vis_params, cmap = param_dict[preset.lower()]

    pt = ee.Geometry.Point(pt)
    region = pt.buffer((patch_size/2)*scale).bounds()

    url = image.getThumbURL({
        'region': region,
        'dimensions': f'{patch_size}x{patch_size}',
        'format': 'png',
        'min': vis_params['min'],
        'max': vis_params['max'],
        'bands': vis_params['bands']
    })

    # Fetch the image from the URL
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to retrieve image.")

    # Convert to image and display
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def get_patch(image: ee.Image,
              pt: tuple[float, float],
              scale: Union[int, tuple[int, int]],
              crs_epsg: str = None,
              patch_size: int = 256,
              add_x_offset: int = 0,
              add_y_offset: int = 0,
              file_format: str = "NUMPY_NDARRAY"):
    """Get a patch centered on the coordinates as a structured numpy array.
    NOTE:
        - coords expects [lon, lat] with coordinate values in WGS84.
        - projection dict is ignored if scale and CRS are supplied.
        - all bands in the input image are retried. Drop bands in input image to select only certain bands."""
    if crs_epsg is None:
        crs_epsg = get_utm_epsg(pt)

    if not isinstance(image, ee.Image):
        raise TypeError("Input image must be an ee.Image object")

    # Unpack scale
    if isinstance(scale, int):
        scale_x, scale_y = scale, -scale
    elif len(scale) == 2:
        scale_x, scale_y = scale[0], -scale[1]
        assert isinstance(scale_x, int) and isinstance(scale_y, int), "Scale values must be integers"
    else:
        raise TypeError("Scale must be an integer or tuple of two integers")

    # Convert WGS84 to UTM
    centroid = wgs84_to_utm(pt, crs_epsg=crs_epsg)
    if centroid is None:
        raise ValueError("UTM conversion failed. Check the coordinate system.")

    # Offset to the upper left corner + any additional offset requested
    offset_x = -scale_x * (patch_size/2) + scale_x * add_x_offset    # move top-left corner right to align centroid
    offset_y = -scale_y * (patch_size/2) + -scale_y * add_y_offset   # but scale_y is flipped so negate to move up

    # Request template
    request = {
        'expression': image,
        'fileFormat': file_format,
        'grid': {
            'dimensions': {
                'width': patch_size,
                'height': patch_size
            },
            'affineTransform': {
                'scaleX': scale_x,
                'shearX': 0,
                'translateX': centroid[0] + offset_x,
                'scaleY': scale_y,
                'shearY': 0,
                'translateY': centroid[1] + offset_y
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


def plot_neighbourhood(patches: Union[np.ndarray, dict],
                       levels: int,
                       bands: Union[str, list],
                       vis_min: float,
                       vis_max: float,
                       **kwargs):
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


class LandsatSR:
    def __init__(self, start_date: str, end_date: str) -> None:
        """
        Args
        - start_date: str, string representation of start date
        - end_date: str, string representation of end date
        """
        self.start_date = start_date
        self.end_date = end_date

        self.l7 = (
            ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
            .filter('WRS_ROW < 122')                            # Remove nighttime images
            .filterDate(self.start_date, self.end_date)         # Filter to desired range
            .linkCollection(                                    # use TOA Thermal Band values due to processing issues
                ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA'),
                linkedBands='B6_VCID_1')
            .map(self.prep_l47)
        )

        self.l8 = (
            ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filter('WRS_ROW < 122')                            # Remove nighttime images
            .filterDate(self.start_date, self.end_date)         # Filter to desired range
            .linkCollection(                                    # use TOA Thermal Band values due to processing issues
                ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA'),
                linkedBands='B10')
            .map(self.prep_l89)
        )

        self.l9 = (
            ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            .filter('WRS_ROW < 122')                            # Remove nighttime images
            .filterDate(self.start_date, self.end_date)         # Filter to desired range
            .linkCollection(                                    # use TOA Thermal Band values due to processing issues
                ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA'),
                linkedBands='B10')
            .map(self.prep_l89)
        )

        # Merge all collections
        self.all = self.l7.merge(self.l8).merge(self.l9)

    @staticmethod
    def get_cloud_mask(image: ee.Image) -> ee.Image:
        """Get mask for non-cloudy pixels for a Landsat image based on QA_PIXEL."""
        qa = image.select(['QA_PIXEL'])
        cloud_shadow_bit_mask = (1 << 3)
        cloud_bit_mask = (1 << 4)
        return qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0).And(qa.bitwiseAnd(cloud_bit_mask).eq(0))

    @staticmethod
    def prep_l47(image: ee.Image) -> ee.Image:
        """Scale and mask clouds for L4/5/7 Collection 2 Surface Reflectance"""

        # Scale bands
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        # thermal_band = image.select('ST_B6').multiply(0.00341802).add(149.0)

        # Replace thermal pixels with corresponding TOA values (SR missing over much of SSA)
        thermal_band = image.select('B6(_VCID_1)?')

        scaled = optical_bands.addBands(thermal_band).select(
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'B6(_VCID_1)?'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']
        )

        # Mask clouds and cloud-shadowed pixels
        mask = LandsatSR.get_cloud_mask(image=image)

        # Put the new bands back into the original image container and mask them.
        return (
            image.select()
            .addBands(scaled)
            .updateMask(mask)
        )

    @staticmethod
    def prep_l89(image: ee.Image) -> ee.Image:

        """Scale and mask clouds for L8/9 Collection 2 Surface Reflectance"""

        # Scale bands
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        # thermal_band = image.select('ST_B10').multiply(0.00341802).add(149.0)

        # Replace thermal pixels with corresponding TOA values (SR missing over much of SSA)
        thermal_band = image.select('B10')

        # Insert the scaled bands back into the original image container.
        scaled = optical_bands.addBands(thermal_band).select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'B10'],
            ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1'],
        )

        # Mask clouds and cloud-shadowed pixels
        mask = LandsatSR.get_cloud_mask(image=image)

        # Put the new bands back into the original image container and mask them.
        return (
            image.select()
            .addBands(scaled)
            .updateMask(mask)
        )
