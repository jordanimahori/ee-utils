import ee


class LandsatSR:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        bands: list[str] | None = None,
        platforms: list[str] | None = None,
        rescale_bands: bool = False,
        mask_clouds: bool = True,
    ) -> None:
        """
        Abstraction over Landsat Collection 2 Level-2 Surface Reflectance products (Landsat 4–9). When multiple sensor
        families are requested (TM/ETM+ and OLI/OLI-2), bands are mapped to a common set of names (BLUE, GREEN, RED,
        NIR, SWIR1, SWIR2) based on approximate spectral correspondence.

        No spectral cross-calibration is applied. Although bands share semantic names, their spectral response functions
        differ slightly across sensors (e.g., OLI vs. TM/ETM+), and sensors differ in native radiometric resolution and
        noise characteristics. As a result, reflectance values within a harmonised band are comparable but not strictly
        identical across platforms.

        Args:
            start_date: String representation of start date.
            end_date: String representation of end date.
            bands (Optional): List of bands to select.
            platforms (Optional): List of Landsat platforms (e.g. ["LANDSAT_8", "LANDSAT_9"]).
            rescale_bands: If true, rescales Landsat bands by USGS recommended values.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.rescale_bands = rescale_bands
        self.mask_clouds = mask_clouds

        sensor_families = {
            "LANDSAT_4": "TM_ETM",
            "LANDSAT_5": "TM_ETM",
            "LANDSAT_7": "TM_ETM",
            "LANDSAT_8": "OLI",
            "LANDSAT_9": "OLI",
        }

        sensor_bands = {
            "TM_ETM": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
            "OLI": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            "HARMONISED": ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        }

        # Get platforms and sensors
        valid_platforms = set(sensor_families.keys())
        if platforms is None:
            self.platforms = valid_platforms
        elif isinstance(platforms, str):
            self.platforms = {platforms.upper()}
        else:
            self.platforms = {p.upper() for p in platforms}

        if not self.platforms.issubset(valid_platforms):
            raise ValueError(
                f"Platforms must be a subset of {valid_platforms}"
            )

        # Harmonise band names if multiple sensor families are requested
        self.sensors = {sensor_families[p] for p in self.platforms}
        self.sensor_family = "HARMONISED" if len(self.sensors) > 1 else next(iter(self.sensors))
        self.available_bands = sensor_bands[self.sensor_family]
        self.bands = bands if bands is not None else self.available_bands
        if not all([b in self.available_bands for b in self.bands]):
            raise ValueError(f"Bands must be a subset of {self.available_bands}")

        def _prep_l47(image: ee.Image) -> ee.Image:
            """Mask clouds, rename bands, and rescale optical bands by EE default (for L4/5/7)."""
            mask = LandsatSR.get_cloud_mask(image) if self.mask_clouds else None
            if self.sensor_family == "HARMONISED":
                image = image.select(
                ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
                )
            image = image.select(self.bands)
            if self.rescale_bands:
                image = image.multiply(0.0000275).add(-0.2)
            if mask is not None:
                image = image.updateMask(mask)
            return image

        def _prep_l89(image: ee.Image) -> ee.Image:
            """Mask clouds, rename bands, and rescale optical bands by EE default (for L8/9)."""
            mask = LandsatSR.get_cloud_mask(image) if self.mask_clouds else None
            if self.sensor_family == "HARMONISED":
                image = image.select(
                    ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                    ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
                )
            image = image.select(self.bands)
            if self.rescale_bands:
                image = image.multiply(0.0000275).add(-0.2)
            if mask is not None:
                image = image.updateMask(mask)
            return image

        self.platform_configs = {
            "LANDSAT_4": {
                "image_collection": "LANDSAT/LT04/C02/T1_L2",
                "prep_function": _prep_l47,
            },
            "LANDSAT_5": {
                "image_collection": "LANDSAT/LT05/C02/T1_L2",
                "prep_function": _prep_l47,
            },
            "LANDSAT_7": {
                "image_collection": "LANDSAT/LE07/C02/T1_L2",
                "prep_function": _prep_l47,
            },
            "LANDSAT_8": {
                "image_collection": "LANDSAT/LC08/C02/T1_L2",
                "prep_function": _prep_l89,
            },
            "LANDSAT_9": {
                "image_collection": "LANDSAT/LC09/C02/T1_L2",
                "prep_function": _prep_l89,
            },
        }

        # Prep and merge ImageCollections for requested platforms
        self.images = ee.ImageCollection([])
        for platform in self.platforms:
            config = self.platform_configs[platform]
            col = (
                ee.ImageCollection(config["image_collection"])
                .filterDate(self.start_date, self.end_date)
                .map(config["prep_function"])
            )
            self.images = self.images.merge(col)
        self.images = self.images.sort("system:time_start")

    @staticmethod
    def get_cloud_mask(image: ee.Image) -> ee.Image:
        qa = image.select("QA_PIXEL")
        fill = 1 << 0
        dilated = 1 << 1
        cirrus = 1 << 2
        cloud = 1 << 3
        shadow = 1 << 4

        return (
            qa.bitwiseAnd(fill).eq(0)
            .And(qa.bitwiseAnd(dilated).eq(0))
            .And(qa.bitwiseAnd(cloud).eq(0))
            .And(qa.bitwiseAnd(shadow).eq(0))
            .And(qa.bitwiseAnd(cirrus).eq(0))
        )
