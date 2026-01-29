import ee


class LandsatSR:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        bands: list[str] | None = None,
        platforms: list[str] | None = None,
        rescale_bands: bool = True,
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
        self.available_platforms = {
            "LANDSAT_4",
            "LANDSAT_5",
            "LANDSAT_7",
            "LANDSAT_8",
            "LANDSAT_9",
        }
        self.available_bands = {
            "BLUE",
            "GREEN",
            "RED",
            "NIR",
            "SWIR1",
            "SWIR2",
            "TEMP1",
        }

        # If no platforms provided, use all available
        if platforms is None:
            self.platforms = self.available_platforms
        elif isinstance(platforms, str):
            self.platforms = {platforms.upper()}
        else:
            self.platforms = {p.upper() for p in platforms}

        # Check arguments
        if not self.platforms.issubset(self.available_platforms):
            raise ValueError(
                f"Platforms must be a subset of {self.available_platforms}"
            )

        # Prep Landsat 4/5/7 with rescaling
        def _prep_l47(image: ee.Image) -> ee.Image:
            """Mask clouds, rename bands, and rescale optical bands by EE default (for L4/5/7)."""
            optical_bands = image.select("SR_B.")
            if self.rescale_bands:
                optical_bands = optical_bands.multiply(0.0000275).add(-0.2)
            thermal_band = image.select("B6(_VCID_1)?")
            scaled = optical_bands.addBands(thermal_band).select(
                ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "B6(_VCID_1)?"],
                ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "TEMP1"],
            )
            mask = LandsatSR.get_cloud_mask(image)
            return image.select([]).addBands(scaled).updateMask(mask)

        # Prep Landsat 8/9 with rescaling
        def _prep_l89(image: ee.Image) -> ee.Image:
            """Mask clouds, rename bands, and rescale optical bands by EE default (for L8/9)."""
            optical_bands = image.select("SR_B.")
            if self.rescale_bands:
                optical_bands = optical_bands.multiply(0.0000275).add(-0.2)
            thermal_band = image.select("B10")
            scaled = optical_bands.addBands(thermal_band).select(
                ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "B10"],
                ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2", "TEMP1"],
            )
            mask = LandsatSR.get_cloud_mask(image)
            return image.select([]).addBands(scaled).updateMask(mask)

        self.platform_configs = {
            "LANDSAT_4": {
                "sr": "LANDSAT/LT04/C02/T1_L2",
                "toa": "LANDSAT/LT04/C02/T1_TOA",
                "thermal_band": "B6",
                "prep_function": _prep_l47,
            },
            "LANDSAT_5": {
                "sr": "LANDSAT/LT05/C02/T1_L2",
                "toa": "LANDSAT/LT05/C02/T1_TOA",
                "thermal_band": "B6",
                "prep_function": _prep_l47,
            },
            "LANDSAT_7": {
                "sr": "LANDSAT/LE07/C02/T1_L2",
                "toa": "LANDSAT/LE07/C02/T1_TOA",
                "thermal_band": "B6_VCID_1",
                "prep_function": _prep_l47,
            },
            "LANDSAT_8": {
                "sr": "LANDSAT/LC08/C02/T1_L2",
                "toa": "LANDSAT/LC08/C02/T1_TOA",
                "thermal_band": "B10",
                "prep_function": _prep_l89,
            },
            "LANDSAT_9": {
                "sr": "LANDSAT/LC09/C02/T1_L2",
                "toa": "LANDSAT/LC09/C02/T1_TOA",
                "thermal_band": "B10",
                "prep_function": _prep_l89,
            },
        }

        # Prep and merge ImageCollections for requested platforms
        self.images = ee.ImageCollection([])
        for platform in self.platforms:
            config = self.platform_configs[platform]
            col = (
                ee.ImageCollection(config["sr"])
                .filterDate(self.start_date, self.end_date)
                .linkCollection(  # join thermal band from TOA ee.ImageCollection
                    ee.ImageCollection(config["toa"]),
                    linkedBands=[config["thermal_band"]],
                )
                .map(
                    config["prep_function"]
                )  # mask low-quality pixels (and optionally rescale)
            )
            # If bands are specified, filter the ee.ImageCollection.
            if self.bands is not None:
                col = col.select(self.bands)
            self.images = self.images.merge(col)

    @staticmethod
    def get_cloud_mask(image: ee.Image) -> ee.Image:
        """Get mask for non-cloudy pixels for a Landsat image based on QA_PIXEL."""
        qa = image.select(["QA_PIXEL"])
        cloud_bit_mask = 1 << 3
        cloud_shadow_bit_mask = 1 << 4
        return (
            qa.bitwiseAnd(cloud_shadow_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cloud_bit_mask).eq(0))
        )
