"""
FLAC-Raster: Experimental Raster to FLAC Converter

Convert GeoTIFF raster data to FLAC audio format with:
- Lossless compression (7-15x size reduction)
- Spatial tiling for HTTP range streaming
- Support for remote files (HTTP, S3, Azure, GCS)

Supports all common satellite imagery data types:
- uint8, int8: 8-bit data (RGB composites, classification)
- uint16, int16: 16-bit data (Sentinel-2, Landsat, DEMs)
- uint32, int32: 32-bit integer data
- float32, float64: Floating point (reflectance, NDVI, processed data)
"""

from .compare import compare_tiffs, display_comparison_table
from .converter import RasterFLACConverter
from .normalization import (
    NormalizationParams,
    calculate_audio_params,
    denormalize_from_audio,
    estimate_precision_loss,
    normalize_to_audio,
)
from .remote import download_remote, is_remote_url, open_remote
from .spatial_encoder import SpatialFLACEncoder, SpatialFLACStreamer, SpatialIndex

# Conditionally import async features
try:
    from .async_reader import (
        ASYNC_GEOTIFF_AVAILABLE,
        AsyncGeoTIFFReader,
        read_geotiff_async,
        read_tile_async,
    )
except ImportError:
    ASYNC_GEOTIFF_AVAILABLE = False
    AsyncGeoTIFFReader = None
    read_geotiff_async = None
    read_tile_async = None

__version__ = "0.2.0"  # Keep in sync with pyproject.toml and pixi.toml
__all__ = [
    # Core converter
    "RasterFLACConverter",
    # Comparison utilities
    "compare_tiffs",
    "display_comparison_table",
    # Spatial encoding
    "SpatialFLACEncoder",
    "SpatialFLACStreamer",
    "SpatialIndex",
    # Normalization
    "normalize_to_audio",
    "denormalize_from_audio",
    "calculate_audio_params",
    "NormalizationParams",
    "estimate_precision_loss",
    # Remote access
    "is_remote_url",
    "open_remote",
    "download_remote",
    # Async (optional)
    "ASYNC_GEOTIFF_AVAILABLE",
    "AsyncGeoTIFFReader",
    "read_geotiff_async",
    "read_tile_async",
]
