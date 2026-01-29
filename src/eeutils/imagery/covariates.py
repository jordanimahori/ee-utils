import ee
import numpy as np


class SpatialCovariates:
    """Generate annual composite covariate layers."""

    TERRACLIMATE_START, TERRACLIMATE_END = 1958, 2024  # IDAHO_EPSCOR/TERRACLIMATE
    GHS_START, GHS_END = 1975, 2030  # JRC/GHSL/P2023A (5-year cadence)
    VIIRS_START, VIIRS_END = 2012, 2024  # NASA/VIIRS/002/VNP46A2
    CHIRPS_START, CHIRPS_END = 1981, 2025  # UCSB-CHG/CHIRPS/PENTAD
    DYNWORLD_START, DYNWORLD_END = 2016, 2025  # GOOGLE/DYNAMICWORLD/V1

    def __init__(
        self,
        year: int,
        landcover_scale: int = 100,
        nodata_val: int = -9999,
        max_pixels: int = 64,
        best_effort: bool = True,
    ):
        self.year = year
        self.cov_years = {
            "terraclimate": int(
                np.clip(self.year, self.TERRACLIMATE_START, self.TERRACLIMATE_END)
            ),
            "ghs": int(np.clip((self.year // 5) * 5, self.GHS_START, self.GHS_END)),
            "viirs": int(np.clip(self.year, self.VIIRS_START, self.VIIRS_END)),
            "chirps": int(np.clip(self.year, self.CHIRPS_START, self.CHIRPS_END)),
            "dynworld": int(np.clip(self.year, self.DYNWORLD_START, self.DYNWORLD_END)),
        }
        self.landcover_scale = landcover_scale
        self.nodata_val = nodata_val
        self.max_pixels = max_pixels
        self.best_effort = best_effort

        terraclimate = (
            ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
            .filterDate(
                f"{self.cov_years['terraclimate']}",
                f"{self.cov_years['terraclimate'] + 1}",
            )
            .select(["tmmx", "tmmn"])
        )
        ghs_built_surfaces = ee.ImageCollection(
            "JRC/GHSL/P2023A/GHS_BUILT_S"
        ).filterDate(f"{self.cov_years['ghs']}", f"{self.cov_years['ghs'] + 1}")
        ghs_population = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").filterDate(
            f"{self.cov_years['ghs']}", f"{self.cov_years['ghs'] + 1}"
        )
        viirs_nightlights = (
            ee.ImageCollection("NASA/VIIRS/002/VNP46A2")
            .filterDate(f"{self.cov_years['viirs']}", f"{self.cov_years['viirs'] + 1}")
            .select(["DNB_BRDF_Corrected_NTL"], ["viirs"])
        )
        precipitation = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(
            f"{self.cov_years['chirps']}", f"{self.cov_years['chirps'] + 1}"
        )
        land_cover = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate(
            f"{self.cov_years['dynworld']}", f"{self.cov_years['dynworld'] + 1}"
        )

        tc_default = terraclimate.first().projection()
        viirs_default = viirs_nightlights.first().projection()
        precip_default = precipitation.first().projection()
        lc_default = ee.Projection("EPSG:4326").atScale(self.landcover_scale)

        reducer_multi = (
            ee.Reducer.mean()
            .combine(ee.Reducer.max(), sharedInputs=True)
            .combine(ee.Reducer.min(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
        )

        self.precip_comp = (
            precipitation.reduce(reducer=reducer_multi)
            .setDefaultProjection(precip_default)
            .reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=self.max_pixels,
                bestEffort=self.best_effort,
            )
            .unmask(self.nodata_val)
        )

        landcover_prob_comp = (
            land_cover.select(
                [
                    "water",
                    "trees",
                    "grass",
                    "flooded_vegetation",
                    "crops",
                    "shrub_and_scrub",
                    "built",
                    "bare",
                    "snow_and_ice",
                ]
            )
            .mean()
            .setDefaultProjection(lc_default)
            .reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=self.max_pixels,
                bestEffort=self.best_effort,
            )
            .unmask(self.nodata_val)
        )
        self.landcover_comp = self._prefix_bands(landcover_prob_comp, "lulc_prob_")

        self.nightlights_comp = (
            viirs_nightlights.reduce(reducer=reducer_multi)
            .setDefaultProjection(viirs_default)
            .reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=self.max_pixels,
                bestEffort=self.best_effort,
            )
            .unmask(self.nodata_val)
        )

        pop_comp = (
            ghs_population.first()
            .reduceResolution(
                reducer=ee.Reducer.sum(),
                maxPixels=self.max_pixels,
                bestEffort=self.best_effort,
            )
            .unmask(self.nodata_val)
        )
        settlement_comp = (
            ghs_built_surfaces.first()
            .reduceResolution(
                reducer=ee.Reducer.sum(),
                maxPixels=self.max_pixels,
                bestEffort=self.best_effort,
            )
            .unmask(self.nodata_val)
        )
        self.ghs_comp = ee.Image.cat([pop_comp, settlement_comp])

        self.terraclimate_comp = (
            terraclimate.reduce(reducer=reducer_multi)
            .setDefaultProjection(tc_default)
            .reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=self.max_pixels,
                bestEffort=self.best_effort,
            )
            .unmask(self.nodata_val)
        )

        self.soil_ph_comp = (
            ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
            .select(["b0"], ["soil_ph_0cm"])
            .reduceResolution(
                reducer=ee.Reducer.mean(),
                maxPixels=self.max_pixels,
                bestEffort=self.best_effort,
            )
            .unmask(self.nodata_val)
        )

        self.all_covariates_comp = ee.Image.cat(
            [
                self.landcover_comp,
                self.nightlights_comp,
                self.ghs_comp,
                self.terraclimate_comp,
                self.precip_comp,
                self.soil_ph_comp,
            ]
        ).toFloat()

    @staticmethod
    def _prefix_bands(img, prefix):
        new_names = img.bandNames().map(lambda b: ee.String(prefix).cat(ee.String(b)))
        return img.rename(new_names)
