"""
Async GeoTIFF reader module for FLAC-Raster.

Provides high-performance async reading of COGs and GeoTIFFs
using async-geotiff with obstore for cloud storage.

This module is optional - install with: pip install flac-raster[async]
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger("flac_raster.async_reader")

# Check for async-geotiff availability
try:
    from async_geotiff import GeoTIFF, Window

    ASYNC_GEOTIFF_AVAILABLE = True
except ImportError:
    ASYNC_GEOTIFF_AVAILABLE = False
    logger.debug("async-geotiff not available - async features disabled")

# Check for obstore availability
try:
    from obstore.store import AzureStore, GCSStore, HTTPStore, LocalStore, S3Store

    OBSTORE_AVAILABLE = True
except ImportError:
    OBSTORE_AVAILABLE = False
    logger.debug("obstore not available - cloud storage features disabled")


def parse_url(url: str) -> Tuple[str, str, str]:
    """
    Parse a URL into (scheme, bucket/host, path).

    Supports:
    - s3://bucket/path/to/file.tif
    - az://container/path/to/file.tif
    - gs://bucket/path/to/file.tif
    - https://host/path/to/file.tif
    - /local/path/to/file.tif
    """
    if url.startswith("/") or "://" not in url:
        # Local file
        return "file", "", url

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    bucket = parsed.netloc
    path = parsed.path.lstrip("/")

    return scheme, bucket, path


def get_store(url: str):
    """
    Create an obstore Store for the given URL.

    Args:
        url: File URL (s3://, az://, gs://, https://, or local path)

    Returns:
        obstore Store instance
    """
    if not OBSTORE_AVAILABLE:
        raise ImportError(
            "obstore is required for async reading. Install with: pip install flac-raster[async]"
        )

    scheme, bucket, path = parse_url(url)

    if scheme == "s3":
        return S3Store(bucket=bucket), path
    elif scheme == "az":
        return AzureStore(container=bucket), path
    elif scheme == "gs":
        return GCSStore(bucket=bucket), path
    elif scheme in ("http", "https"):
        return HTTPStore(base_url=f"{scheme}://{bucket}"), path
    elif scheme == "file":
        # Local file - use LocalStore with directory as root
        file_path = Path(path)
        return LocalStore(prefix=str(file_path.parent)), file_path.name
    else:
        raise ValueError(f"Unsupported URL scheme: {scheme}")


async def read_geotiff_async(
    url: str,
    window: Optional[Tuple[int, int, int, int]] = None,
    bands: Optional[list] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Async read a GeoTIFF file.

    Args:
        url: File URL (local, HTTP, S3, Azure, GCS)
        window: Optional (col_off, row_off, width, height) to read subset
        bands: Optional list of band indices to read (0-indexed)

    Returns:
        (data_array, metadata_dict)
    """
    if not ASYNC_GEOTIFF_AVAILABLE:
        raise ImportError(
            "async-geotiff is required for async reading. "
            "Install with: pip install flac-raster[async]"
        )

    store, path = get_store(url)
    geotiff = await GeoTIFF.open(path, store=store)

    # Build metadata
    metadata = {
        "width": geotiff.width,
        "height": geotiff.height,
        "count": geotiff.count,
        "crs": str(geotiff.crs) if geotiff.crs else None,
        "transform": list(geotiff.transform) if geotiff.transform else None,
        "dtype": str(geotiff.dtype),
    }

    # Read data
    if window:
        col_off, row_off, width, height = window
        read_window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
        data = await geotiff.read(window=read_window, bands=bands)
    else:
        data = await geotiff.read(bands=bands)

    return data, metadata


async def read_tile_async(
    url: str,
    tile_x: int,
    tile_y: int,
    tile_size: int = 256,
) -> Tuple[np.ndarray, dict]:
    """
    Async read a specific tile from a COG.

    Args:
        url: File URL
        tile_x: Tile X index
        tile_y: Tile Y index
        tile_size: Tile size in pixels (default: 256)

    Returns:
        (tile_data, metadata)
    """
    col_off = tile_x * tile_size
    row_off = tile_y * tile_size

    return await read_geotiff_async(
        url,
        window=(col_off, row_off, tile_size, tile_size),
    )


class AsyncGeoTIFFReader:
    """
    Async context manager for reading GeoTIFF files.

    Example:
        async with AsyncGeoTIFFReader("s3://bucket/file.tif") as reader:
            data = await reader.read_window(0, 0, 512, 512)
            print(reader.metadata)
    """

    def __init__(self, url: str):
        """
        Initialize async reader.

        Args:
            url: File URL (local, HTTP, S3, Azure, GCS)
        """
        if not ASYNC_GEOTIFF_AVAILABLE:
            raise ImportError(
                "async-geotiff is required. Install with: pip install flac-raster[async]"
            )

        self.url = url
        self._geotiff = None
        self._store = None
        self._path = None

    async def __aenter__(self):
        """Open the GeoTIFF file."""
        self._store, self._path = get_store(self.url)
        self._geotiff = await GeoTIFF.open(self._path, store=self._store)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close resources."""
        self._geotiff = None
        self._store = None
        return False

    @property
    def metadata(self) -> dict:
        """Get file metadata."""
        if not self._geotiff:
            raise RuntimeError("Reader not opened. Use 'async with' context.")

        return {
            "width": self._geotiff.width,
            "height": self._geotiff.height,
            "count": self._geotiff.count,
            "crs": str(self._geotiff.crs) if self._geotiff.crs else None,
            "transform": list(self._geotiff.transform) if self._geotiff.transform else None,
            "dtype": str(self._geotiff.dtype),
        }

    @property
    def width(self) -> int:
        return self._geotiff.width

    @property
    def height(self) -> int:
        return self._geotiff.height

    @property
    def count(self) -> int:
        return self._geotiff.count

    async def read(self, bands: Optional[list] = None) -> np.ndarray:
        """Read full raster data."""
        if not self._geotiff:
            raise RuntimeError("Reader not opened. Use 'async with' context.")
        return await self._geotiff.read(bands=bands)

    async def read_window(
        self,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
        bands: Optional[list] = None,
    ) -> np.ndarray:
        """Read a window of data."""
        if not self._geotiff:
            raise RuntimeError("Reader not opened. Use 'async with' context.")

        window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
        return await self._geotiff.read(window=window, bands=bands)

    async def read_tile(
        self,
        tile_x: int,
        tile_y: int,
        tile_size: int = 256,
        bands: Optional[list] = None,
    ) -> np.ndarray:
        """Read a tile by tile coordinates."""
        return await self.read_window(
            col_off=tile_x * tile_size,
            row_off=tile_y * tile_size,
            width=tile_size,
            height=tile_size,
            bands=bands,
        )
