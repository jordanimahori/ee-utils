from pyproj import Transformer


def get_utm_epsg(pt: tuple[float, float]) -> str:
    """Determine the UTM EPSG code for a given lon, lat pair in WGS84."""
    lon, lat = pt
    if not (-180 <= lon <= 180 and -80 <= lat <= 84):
        raise ValueError("UTM only defined for latitudes between -80 and 84 degrees")
    lon = 179.999999 if lon == 180 else lon  # avoid zone=61 at the antimeridian
    utm_zone = int((lon + 180) // 6) + 1  # UTM zone number (ignoring Svalbard)
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone  # 326xx for NH and 327xx for SH
    return f"EPSG:{epsg_code}"


def wgs84_to_utm(pt: tuple[float, float], crs_epsg: str) -> tuple[float, float]:
    """Convert lon, lat (WGS84) to UTM coordinates in the requested zone."""
    transformer = Transformer.from_crs(crs_from="epsg:4326", crs_to=crs_epsg, always_xy=True)
    return transformer.transform(*pt)
