"""
Spatial FLAC Encoder with per-frame bbox metadata support
Enables HTTP range request streaming for geospatial data

Supports all common satellite imagery data types:
- uint8, int8: 8-bit data (RGB composites, classification)
- uint16, int16: 16-bit data (Sentinel-2, Landsat, DEMs)
- uint32, int32: 32-bit integer data
- float32, float64: Floating point (reflectance, NDVI, processed data)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyflac
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window
from rich.console import Console

from .normalization import (
    NormalizationParams,
    calculate_audio_params,
    normalize_to_audio,
)

console = Console()


class SpatialFrame:
    """Represents a spatial FLAC frame with bbox metadata"""

    def __init__(
        self,
        frame_id: int,
        bbox: Tuple[float, float, float, float],
        window: Window,
        byte_offset: int = 0,
        byte_size: int = 0,
    ):
        self.frame_id = frame_id
        self.bbox = bbox  # (xmin, ymin, xmax, ymax)
        self.window = window  # Rasterio window (row_off, col_off, height, width)
        self.byte_offset = byte_offset
        self.byte_size = byte_size

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "frame_id": self.frame_id,
            "bbox": self.bbox,
            "window": {
                "row_off": self.window.row_off,
                "col_off": self.window.col_off,
                "height": self.window.height,
                "width": self.window.width,
            },
            "byte_offset": self.byte_offset,
            "byte_size": self.byte_size,
        }


class SpatialIndex:
    """Spatial index for FLAC frames with bbox lookup"""

    def __init__(self, frames: List[SpatialFrame], crs: CRS, transform: Affine):
        self.frames = frames
        self.crs = crs
        self.transform = transform
        self.total_bytes = sum(frame.byte_size for frame in frames)

    def query_bbox(self, bbox: Tuple[float, float, float, float]) -> List[SpatialFrame]:
        """Find frames that intersect with given bbox"""
        intersecting = []
        xmin, ymin, xmax, ymax = bbox

        for frame in self.frames:
            fxmin, fymin, fxmax, fymax = frame.bbox

            # Check if bboxes intersect
            if xmin < fxmax and xmax > fxmin and ymin < fymax and ymax > fymin:
                intersecting.append(frame)

        return intersecting

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "crs": str(self.crs),
            "transform": list(self.transform),
            "frames": [frame.to_dict() for frame in self.frames],
        }


class SpatialFLACEncoder:
    """Enhanced FLAC encoder with spatial tiling and bbox metadata"""

    def __init__(self, tile_size: int = 512):
        self.tile_size = tile_size
        self.logger = logging.getLogger("flac_raster.spatial_encoder")
        self.frames: List[SpatialFrame] = []
        self.current_frame_id = 0
        self.bytes_written = 0
        self.output_file = None

    def _calculate_tiles(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """Calculate tile boundaries for spatial partitioning"""
        tiles = []

        for row_start in range(0, height, self.tile_size):
            for col_start in range(0, width, self.tile_size):
                row_end = min(row_start + self.tile_size, height)
                col_end = min(col_start + self.tile_size, width)
                tiles.append((row_start, col_start, row_end - row_start, col_end - col_start))

        self.logger.info(f"Created {len(tiles)} tiles of size {self.tile_size}x{self.tile_size}")
        return tiles

    def _tile_to_bbox(
        self, row_off: int, col_off: int, height: int, width: int, transform: Affine
    ) -> Tuple[float, float, float, float]:
        """Convert tile pixel coordinates to geographic bbox"""
        # Transform pixel coordinates to geographic coordinates
        xmin, ymax = transform * (col_off, row_off)
        xmax, ymin = transform * (col_off + width, row_off + height)

        return (xmin, ymin, xmax, ymax)

    def _write_callback(self, buffer: bytes, num_bytes: int, num_samples: int, current_frame: int):
        """Custom write callback that tracks frame metadata"""
        if self.output_file:
            self.output_file.write(buffer)

            # Update frame metadata with byte position
            if current_frame < len(self.frames):
                frame = self.frames[current_frame]
                if frame.byte_offset == 0:  # First write for this frame
                    frame.byte_offset = self.bytes_written
                frame.byte_size += num_bytes

            self.bytes_written += num_bytes

        return 0  # Success

    def _metadata_callback(self, metadata):
        """Handle metadata writing for spatial frames"""
        self.logger.debug(f"Metadata callback called for frame {self.current_frame_id}")
        # In a full implementation, we would write APPLICATION metadata blocks here
        # For now, we'll store metadata in sidecar files

    def encode_spatial_flac(
        self,
        tiff_path: Path,
        flac_path: Path,
        compression_level: int = 5,
        enable_streaming: bool = True,
    ) -> SpatialIndex:
        """
        Encode TIFF to FLAC with spatial tiling and bbox metadata

        Args:
            tiff_path: Input TIFF file path
            flac_path: Output FLAC file path
            compression_level: FLAC compression level (0-8)
            enable_streaming: Whether to enable HTTP range streaming metadata

        Returns:
            SpatialIndex: Spatial index with frame metadata
        """
        self.logger.info(f"Starting spatial FLAC encoding: {tiff_path} -> {flac_path}")

        # Read raster data
        with rasterio.open(tiff_path) as src:
            # Read full raster
            raster_data = src.read()
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width

            self.logger.info(f"Raster: {height}x{width}, {src.count} bands, {src.dtypes[0]}")

        # Calculate tiles
        tiles = self._calculate_tiles(height, width)

        # Prepare output
        with open(flac_path, "wb") as f:
            self.output_file = f
            self.bytes_written = 0
            self.frames = []

            # Process each tile as a separate FLAC "track"
            for i, (row_off, col_off, tile_height, tile_width) in enumerate(tiles):
                window = Window(col_off, row_off, tile_width, tile_height)
                bbox = self._tile_to_bbox(row_off, col_off, tile_height, tile_width, transform)

                # Create frame metadata
                frame = SpatialFrame(i, bbox, window)
                self.frames.append(frame)

                # Read tile data
                with rasterio.open(tiff_path) as src:
                    tile_data = src.read(window=window)

                # Reshape for FLAC encoding (channels, samples)
                if tile_data.ndim == 3:  # Multi-band
                    bands, tile_h, tile_w = tile_data.shape
                    reshaped_data = tile_data.reshape(bands, tile_h * tile_w).T
                else:  # Single band
                    reshaped_data = tile_data.flatten().reshape(-1, 1)

                # Calculate audio parameters using unified module
                sample_rate, bits_per_sample = self._calculate_tile_audio_params(
                    tile_data, tile_data.dtype
                )
                channels = reshaped_data.shape[1]

                # Normalize to audio range using unified normalization
                audio_data, _ = self._normalize_tile_to_audio(reshaped_data, bits_per_sample)

                self.logger.debug(f"Tile {i}: {tile_height}x{tile_width}, bbox: {bbox}")

                # Create FLAC encoder for this tile
                # Note: Each tile becomes a separate FLAC "song" in the file
                # This is a simplified approach - a full implementation would use
                # FLAC application metadata blocks to store bbox info per frame

                # For now, we'll create a concatenated FLAC file where each tile
                # is a separate complete FLAC stream with its own header
                tile_flac_data = self._encode_tile_to_flac(
                    audio_data, sample_rate, channels, bits_per_sample, compression_level
                )

                # Write tile data
                frame.byte_offset = self.bytes_written
                f.write(tile_flac_data)
                frame.byte_size = len(tile_flac_data)
                self.bytes_written += len(tile_flac_data)

                self.logger.debug(
                    f"Encoded tile {i}: {frame.byte_size} bytes at offset {frame.byte_offset}"
                )

        # Create spatial index
        spatial_index = SpatialIndex(self.frames, crs, transform)

        # Embed all metadata directly in FLAC file using VORBIS_COMMENT
        self._embed_metadata_in_flac(
            flac_path, spatial_index, crs, transform, height, width, raster_data, tiles
        )

        console.print(
            f"[green]SUCCESS: Encoded {len(tiles)} spatial tiles to FLAC: {flac_path}[/green]"
        )
        return spatial_index

    def _normalize_tile_to_audio(
        self, data: np.ndarray, bits_per_sample: int
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """Normalize tile data to audio range using unified normalization."""
        return normalize_to_audio(data, bits_per_sample)

    def _calculate_tile_audio_params(
        self, tile_data: np.ndarray, dtype: np.dtype
    ) -> Tuple[int, int]:
        """Calculate sample rate and bit depth using unified calculation."""
        return calculate_audio_params(tile_data, dtype)

    def _encode_tile_to_flac(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        channels: int,
        bits_per_sample: int,
        compression_level: int,
    ) -> bytes:
        """Encode a single tile to FLAC bytes"""
        from io import BytesIO

        # Create output buffer
        output_buffer = BytesIO()

        def write_callback(buffer, num_bytes, num_samples, current_frame):
            output_buffer.write(buffer)
            return 0

        # Create encoder
        encoder = pyflac.StreamEncoder(
            write_callback=write_callback,
            sample_rate=sample_rate,
            compression_level=compression_level,
            blocksize=4096,
        )

        # Set encoder parameters
        encoder._channels = channels
        encoder._bits_per_sample = bits_per_sample

        # Encode data
        encoder.process(audio_data)
        encoder.finish()

        # Return encoded bytes
        return output_buffer.getvalue()

    def _embed_metadata_in_flac(
        self,
        flac_path: Path,
        spatial_index: SpatialIndex,
        crs: CRS,
        transform: Affine,
        height: int,
        width: int,
        raster_data: np.ndarray,
        tiles: List,
    ):
        """Embed all metadata directly in FLAC file using mutagen"""
        try:
            from mutagen.flac import FLAC

            # Open FLAC file for metadata editing
            flac_file = FLAC(str(flac_path))

            # Clear existing comments
            flac_file.clear()

            # Standard FLAC metadata
            flac_file["TITLE"] = "Geospatial Raster Data"
            flac_file["DESCRIPTION"] = (
                f"TIFF raster converted to spatial FLAC with {len(tiles)} tiles"
            )
            flac_file["ENCODER"] = "FLAC-Raster v0.1.0"
            flac_file["DATE"] = str(np.datetime64("now", "D"))

            # Core geospatial metadata
            flac_file["GEOSPATIAL_CRS"] = str(crs)
            flac_file["GEOSPATIAL_WIDTH"] = str(width)
            flac_file["GEOSPATIAL_HEIGHT"] = str(height)
            flac_file["GEOSPATIAL_COUNT"] = str(raster_data.shape[0])
            flac_file["GEOSPATIAL_DTYPE"] = str(raster_data.dtype)
            flac_file["GEOSPATIAL_DATA_MIN"] = str(float(np.min(raster_data)))
            flac_file["GEOSPATIAL_DATA_MAX"] = str(float(np.max(raster_data)))

            # Transform as JSON string
            flac_file["GEOSPATIAL_TRANSFORM"] = json.dumps(list(transform))

            # Bounds - correctly calculate from all frames (not assuming order)
            all_bboxes = [f.bbox for f in spatial_index.frames]
            bounds = [
                min(b[0] for b in all_bboxes),  # xmin
                min(b[1] for b in all_bboxes),  # ymin
                max(b[2] for b in all_bboxes),  # xmax
                max(b[3] for b in all_bboxes),  # ymax
            ]
            flac_file["GEOSPATIAL_BOUNDS"] = json.dumps(bounds)

            # Spatial tiling metadata
            flac_file["GEOSPATIAL_SPATIAL_TILING"] = "true"
            flac_file["GEOSPATIAL_TILE_SIZE"] = str(self.tile_size)
            flac_file["GEOSPATIAL_NUM_TILES"] = str(len(tiles))

            # Embed complete spatial index as compressed JSON
            spatial_data = spatial_index.to_dict()
            spatial_json = json.dumps(spatial_data, separators=(",", ":"))

            # Compress spatial index for efficiency
            import base64
            import gzip

            compressed = gzip.compress(spatial_json.encode("utf-8"))
            encoded = base64.b64encode(compressed).decode("ascii")
            flac_file["GEOSPATIAL_SPATIAL_INDEX"] = encoded

            # Save metadata to FLAC file
            flac_file.save()

            self.logger.info(
                "[SUCCESS] Embedded complete metadata in FLAC file (no sidecar needed)"
            )
            console.print(
                "[green][SUCCESS] All metadata embedded in FLAC file - no sidecar files needed![/green]"
            )

        except ImportError:
            self.logger.warning("mutagen not available - falling back to JSON sidecar")
            console.print(
                "[yellow][WARNING] Install mutagen for embedded metadata: pip install mutagen[/yellow]"
            )

            # Fallback to JSON sidecar
            index_path = flac_path.with_suffix(".spatial.json")
            with open(index_path, "w") as f:
                json.dump(spatial_index.to_dict(), f, indent=2)
            self.logger.info(f"Spatial index saved as sidecar: {index_path}")

        except Exception as e:
            self.logger.error(f"Failed to embed metadata: {e}")
            console.print(f"[red][ERROR] Failed to embed metadata: {e}[/red]")

            # Fallback to JSON sidecar
            index_path = flac_path.with_suffix(".spatial.json")
            with open(index_path, "w") as f:
                json.dump(spatial_index.to_dict(), f, indent=2)
            self.logger.info(f"Spatial index saved as fallback sidecar: {index_path}")


class SpatialFLACStreamer:
    """HTTP range request streaming for spatial FLAC files.

    Supports:
    - Local files
    - HTTP/HTTPS URLs
    - S3 URLs (s3://bucket/key) via obstore
    - Azure Blob (az://container/blob) via obstore
    - Google Cloud Storage (gs://bucket/key) via obstore
    """

    def __init__(self, flac_path):
        self.flac_path = flac_path
        self.is_remote = isinstance(flac_path, str) and self._is_remote_url(flac_path)
        # Keep is_url for backward compatibility
        self.is_url = self.is_remote
        self.logger = logging.getLogger("flac_raster.spatial_streamer")
        self._remote_file = None
        self.spatial_index = self._load_spatial_index()

    def _is_remote_url(self, path: str) -> bool:
        """Check if path is a remote URL (HTTP, S3, Azure, GCS)."""
        return path.startswith(("http://", "https://", "s3://", "az://", "gs://"))

    def _load_spatial_index(self) -> SpatialIndex:
        """Load spatial index from embedded FLAC metadata or fallback to sidecar file"""

        # First try to read from embedded FLAC metadata
        try:
            from mutagen.flac import FLAC

            if self.is_remote:
                # Download FLAC metadata from remote URL
                import tempfile

                from .remote import RemoteFile

                self._remote_file = RemoteFile(self.flac_path)

                # Download first 1MB to get FLAC metadata (metadata is at the beginning)
                metadata_bytes = self._remote_file.read_range(0, 1048575)

                # Save to temporary file for mutagen
                with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as temp_file:
                    temp_file.write(metadata_bytes)
                    temp_path = temp_file.name

                try:
                    flac_file = FLAC(temp_path)
                finally:
                    Path(temp_path).unlink()
            else:
                flac_file = FLAC(str(self.flac_path))

            if "GEOSPATIAL_SPATIAL_INDEX" in flac_file:
                self.logger.info("Reading spatial index from embedded FLAC metadata")

                # Decode compressed spatial index
                import base64
                import gzip

                encoded = flac_file["GEOSPATIAL_SPATIAL_INDEX"][0]
                compressed = base64.b64decode(encoded.encode("ascii"))
                spatial_json = gzip.decompress(compressed).decode("utf-8")
                data = json.loads(spatial_json)

            else:
                raise ValueError("No embedded spatial index found")

        except (ImportError, Exception) as e:
            self.logger.warning(f"Failed to read embedded metadata: {e}")
            self.logger.info("Falling back to sidecar file")

            # Fallback to sidecar file
            index_path = self.flac_path.with_suffix(".spatial.json")

            if not index_path.exists():
                raise FileNotFoundError(
                    f"Spatial index not found in FLAC metadata or sidecar file: {index_path}"
                )

            with open(index_path, "r") as f:
                data = json.load(f)

        # Reconstruct frames
        frames = []
        for frame_data in data["frames"]:
            window = Window(
                frame_data["window"]["col_off"],
                frame_data["window"]["row_off"],
                frame_data["window"]["width"],
                frame_data["window"]["height"],
            )
            frame = SpatialFrame(
                frame_data["frame_id"],
                tuple(frame_data["bbox"]),
                window,
                frame_data["byte_offset"],
                frame_data["byte_size"],
            )
            frames.append(frame)

        crs = CRS.from_string(data["crs"])
        transform = Affine(*data["transform"])

        return SpatialIndex(frames, crs, transform)

    def get_byte_ranges_for_bbox(
        self, bbox: Tuple[float, float, float, float]
    ) -> List[Tuple[int, int]]:
        """Get HTTP byte ranges for frames intersecting bbox"""
        intersecting_frames = self.spatial_index.query_bbox(bbox)

        ranges = []
        for frame in intersecting_frames:
            if frame.byte_size > 0:
                end_byte = frame.byte_offset + frame.byte_size - 1
                ranges.append((frame.byte_offset, end_byte))

        # Sort by offset and merge overlapping ranges
        ranges.sort()
        merged = []
        for start, end in ranges:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        self.logger.info(f"Found {len(merged)} byte ranges for bbox {bbox}")
        return merged

    def stream_bbox_data(self, bbox: Tuple[float, float, float, float]) -> bytes:
        """Stream FLAC data for specific bbox using range requests.

        Supports local files and remote URLs (HTTP, S3, Azure, GCS).
        """
        ranges = self.get_byte_ranges_for_bbox(bbox)

        data_chunks = []
        if self.is_remote:
            # Use remote file handler for all remote URLs
            from .remote import RemoteFile

            if self._remote_file is None:
                self._remote_file = RemoteFile(self.flac_path)

            for start, end in ranges:
                chunk = self._remote_file.read_range(start, end)
                data_chunks.append(chunk)
        else:
            # Local file
            with open(self.flac_path, "rb") as f:
                for start, end in ranges:
                    f.seek(start)
                    chunk = f.read(end - start + 1)
                    data_chunks.append(chunk)

        return b"".join(data_chunks)
