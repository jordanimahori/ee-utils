import ee

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