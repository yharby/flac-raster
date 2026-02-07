"""
Core converter functionality for FLAC-Raster

Handles bidirectional conversion between GeoTIFF raster data and FLAC audio format.
Supports all common satellite imagery data types including:
- uint8, int8: 8-bit data (RGB composites, classification)
- uint16, int16: 16-bit data (Sentinel-2, Landsat, DEMs)
- uint32, int32: 32-bit integer data
- float32, float64: Floating point (reflectance, NDVI, processed data)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyflac
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rich.console import Console

from .normalization import (
    NormalizationParams,
    calculate_audio_params,
    denormalize_from_audio,
    normalize_to_audio,
)

console = Console()


class RasterFLACConverter:
    """Handles conversion between TIFF and FLAC formats for raster data"""

    def __init__(self):
        self.metadata_key = "RASTER_METADATA"
        self.logger = logging.getLogger("flac_raster.converter")

    def tiff_to_flac(
        self,
        tiff_path: Path,
        flac_path: Path,
        compression_level: int = 5,
        spatial_tiling: bool = False,
        tile_size: int = 512,
    ):
        """Convert TIFF raster to FLAC format

        Args:
            tiff_path: Input TIFF file path
            flac_path: Output FLAC file path
            compression_level: FLAC compression level (0-8)
            spatial_tiling: Enable spatial tiling for HTTP range streaming
            tile_size: Size of spatial tiles (default 512x512)
        """
        self.logger.info(f"Starting TIFF to FLAC conversion: {tiff_path} -> {flac_path}")
        self.logger.info(f"Compression level: {compression_level}")
        if spatial_tiling:
            self.logger.info(f"Spatial tiling enabled: {tile_size}x{tile_size} tiles")
            console.print("[cyan]Using spatial tiling for HTTP range streaming[/cyan]")
        console.print(f"[cyan]Reading TIFF file: {tiff_path}[/cyan]")

        # Use spatial encoder if spatial tiling is enabled
        if spatial_tiling:
            from .spatial_encoder import SpatialFLACEncoder

            encoder = SpatialFLACEncoder(tile_size=tile_size)
            spatial_index = encoder.encode_spatial_flac(tiff_path, flac_path, compression_level)
            return spatial_index

        with rasterio.open(tiff_path) as src:
            # Read raster data and metadata
            self.logger.debug("Reading raster data")
            data = src.read()
            meta = src.meta.copy()
            bounds = src.bounds
            crs = src.crs

            self.logger.info(f"Raster shape: {data.shape}")
            self.logger.info(f"Raster dtype: {data.dtype}")
            self.logger.info(f"CRS: {crs}")
            self.logger.debug(f"Bounds: {bounds}")
            self.logger.debug(f"Transform: {src.transform}")

            console.print(
                f"[green]Raster info: {meta['width']}x{meta['height']}, "
                f"{meta['count']} band(s), dtype: {meta['dtype']}[/green]"
            )

            # Calculate audio parameters using unified module
            sample_rate, bits_per_sample = calculate_audio_params(data, data.dtype)
            console.print(
                f"[yellow]Using sample rate: {sample_rate}Hz, bit depth: {bits_per_sample}[/yellow]"
            )

            # Reshape data for FLAC (interleave bands as channels)
            if data.ndim == 3:
                # Multiple bands: (bands, height, width) -> (height*width, bands)
                channels = data.shape[0]
                data_reshaped = data.transpose(1, 2, 0).reshape(-1, channels)
                self.logger.info(f"Reshaped multi-band data: {data.shape} -> {data_reshaped.shape}")
            else:
                # Single band: (height, width) -> (height*width, 1)
                channels = 1
                data_reshaped = data.reshape(-1, 1)
                self.logger.info(
                    f"Reshaped single-band data: {data.shape} -> {data_reshaped.shape}"
                )

            # Normalize to audio range using unified normalization
            audio_data, norm_params = normalize_to_audio(data_reshaped, bits_per_sample)
            self.logger.info(f"Data range: [{norm_params.data_min}, {norm_params.data_max}]")

            # Prepare metadata for FLAC (include normalization params for perfect reconstruction)
            raster_metadata = {
                "width": meta["width"],
                "height": meta["height"],
                "count": meta["count"],  # number of bands
                "dtype": str(meta["dtype"]),
                "crs": crs.to_string() if crs else None,
                "transform": list(src.transform) if src.transform else None,
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "data_min": norm_params.data_min,
                "data_max": norm_params.data_max,
                "nodata": meta.get("nodata"),
                "driver": meta["driver"],
                "scale_factor": norm_params.scale_factor,
            }

            # Create FLAC encoder
            self.logger.info(f"Creating FLAC encoder: channels={channels}, blocksize=4096")
            encoder = pyflac.StreamEncoder(
                write_callback=self._get_write_callback(flac_path),
                sample_rate=sample_rate,
                compression_level=compression_level,
                blocksize=4096,  # Use frames for chunking
            )

            # Set channels and bits per sample as attributes
            encoder._channels = channels
            encoder._bits_per_sample = bits_per_sample

            # Process and encode data
            console.print("[cyan]Encoding to FLAC...[/cyan]")
            self.logger.info(f"Processing {audio_data.shape[0]:,} samples")
            encoder.process(audio_data)
            encoder.finish()

            # Close the output file
            if hasattr(self, "output_file"):
                self.output_file.close()

            # Embed metadata directly in FLAC file (after encoding is complete)
            self._embed_metadata_in_flac(flac_path, raster_metadata)

            output_size = flac_path.stat().st_size
            input_size = tiff_path.stat().st_size
            compression_ratio = (1 - output_size / input_size) * 100

            self.logger.info(f"Conversion complete: {output_size / 1024 / 1024:.2f} MB")
            self.logger.info(f"Compression ratio: {compression_ratio:.1f}%")
            console.print(f"[green]SUCCESS: Converted to FLAC: {flac_path}[/green]")
            console.print(
                f"[dim]File size: {output_size / 1024 / 1024:.2f} MB (compression: {compression_ratio:.1f}%)[/dim]"
            )

    def flac_to_tiff(self, flac_path: Path, tiff_path: Path):
        """Convert FLAC back to TIFF format"""
        self.logger.info(f"Starting FLAC to TIFF conversion: {flac_path} -> {tiff_path}")
        console.print(f"[cyan]Reading FLAC file: {flac_path}[/cyan]")

        # Decode FLAC
        self.logger.debug("Creating FLAC decoder")
        decoder = pyflac.FileDecoder(str(flac_path))
        audio_data, sample_rate = decoder.process()
        self.logger.info(
            f"Decoded audio shape: {audio_data.shape}, dtype: {audio_data.dtype}, sample_rate: {sample_rate}"
        )

        # Load metadata from embedded FLAC metadata or fallback to sidecar
        self.logger.info("Loading metadata from FLAC file")
        metadata = self._read_embedded_metadata(flac_path)

        if not metadata:
            self.logger.error("No metadata found in FLAC file or sidecar file")
            raise ValueError("No metadata found in FLAC file or sidecar file")

        self.logger.debug(f"Found metadata: {list(metadata.keys())}")

        console.print(
            f"[green]Found raster metadata: {metadata['width']}x{metadata['height']}, "
            f"{metadata['count']} band(s)[/green]"
        )

        # Reshape audio data back to raster format
        width = metadata["width"]
        height = metadata["height"]
        count = metadata["count"]

        self.logger.info(f"Reshaping to raster: {width}x{height}, {count} band(s)")

        if count > 1:
            # Multiple bands: (height*width, bands) -> (bands, height, width)
            data_reshaped = audio_data.reshape(height, width, count)
            raster_data = data_reshaped.transpose(2, 0, 1)
            self.logger.debug(f"Reshaped multi-band: {audio_data.shape} -> {raster_data.shape}")
        else:
            # Single band: (height*width,) -> (height, width)
            raster_data = audio_data.reshape(height, width)
            self.logger.debug(f"Reshaped single-band: {audio_data.shape} -> {raster_data.shape}")

        # Denormalize from audio range using unified normalization
        original_dtype = np.dtype(metadata["dtype"])
        norm_params = NormalizationParams(
            data_min=metadata["data_min"],
            data_max=metadata["data_max"],
            original_dtype=str(original_dtype),
            bits_per_sample=16 if raster_data.dtype == np.int16 else 24,
            scale_factor=metadata.get(
                "scale_factor", 32767 if raster_data.dtype == np.int16 else 8388607
            ),
        )
        denormalized_data = denormalize_from_audio(raster_data, norm_params)

        # Prepare metadata for rasterio
        meta = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": count,
            "dtype": original_dtype,
            "nodata": metadata.get("nodata"),
        }

        if metadata.get("crs"):
            meta["crs"] = CRS.from_string(metadata["crs"])

        if metadata.get("transform"):
            t = metadata["transform"]
            meta["transform"] = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

        # Write TIFF
        console.print("[cyan]Writing TIFF file...[/cyan]")
        self.logger.info(f"Writing TIFF with metadata: {meta}")

        with rasterio.open(tiff_path, "w", **meta) as dst:
            if count == 1:
                dst.write(denormalized_data, 1)
            else:
                dst.write(denormalized_data)

        output_size = tiff_path.stat().st_size
        self.logger.info(f"TIFF written successfully: {output_size / 1024 / 1024:.2f} MB")
        console.print(f"[green]SUCCESS: Converted to TIFF: {tiff_path}[/green]")

    def _embed_metadata_in_flac(self, flac_path: Path, metadata: Dict):
        """Embed geospatial metadata directly in FLAC file"""
        try:
            from mutagen.flac import FLAC

            # Open FLAC file for metadata editing
            flac_file = FLAC(str(flac_path))

            # Clear existing comments
            flac_file.clear()

            # Standard FLAC metadata
            flac_file["TITLE"] = "Geospatial Raster Data"
            flac_file["DESCRIPTION"] = "TIFF raster converted to FLAC with geospatial metadata"
            flac_file["ENCODER"] = "FLAC-Raster v0.1.0"

            # Core geospatial metadata
            flac_file["GEOSPATIAL_CRS"] = str(metadata.get("crs", ""))
            flac_file["GEOSPATIAL_WIDTH"] = str(metadata.get("width", 0))
            flac_file["GEOSPATIAL_HEIGHT"] = str(metadata.get("height", 0))
            flac_file["GEOSPATIAL_COUNT"] = str(metadata.get("count", 1))
            flac_file["GEOSPATIAL_DTYPE"] = str(metadata.get("dtype", ""))
            flac_file["GEOSPATIAL_NODATA"] = str(metadata.get("nodata", ""))
            flac_file["GEOSPATIAL_DATA_MIN"] = str(metadata.get("data_min", ""))
            flac_file["GEOSPATIAL_DATA_MAX"] = str(metadata.get("data_max", ""))

            # Transform and bounds as JSON
            flac_file["GEOSPATIAL_TRANSFORM"] = json.dumps(metadata.get("transform", []))
            flac_file["GEOSPATIAL_BOUNDS"] = json.dumps(metadata.get("bounds", []))

            # Spatial tiling info
            flac_file["GEOSPATIAL_SPATIAL_TILING"] = str(metadata.get("spatial_tiling", False))

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
            metadata_json = json.dumps(metadata, indent=2)
            metadata_path = flac_path.with_suffix(".json")
            self.logger.info(f"Saving metadata to: {metadata_path}")
            with open(metadata_path, "w") as f:
                f.write(metadata_json)

        except Exception as e:
            self.logger.error(f"Failed to embed metadata: {e}")
            console.print(f"[red][ERROR] Failed to embed metadata: {e}[/red]")

            # Fallback to JSON sidecar
            metadata_json = json.dumps(metadata, indent=2)
            metadata_path = flac_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                f.write(metadata_json)

    def _read_embedded_metadata(self, flac_path: Path) -> Optional[Dict]:
        """Read embedded metadata from FLAC file"""
        try:
            from mutagen.flac import FLAC

            flac_file = FLAC(str(flac_path))

            if "GEOSPATIAL_CRS" in flac_file:
                self.logger.info("Reading metadata from embedded FLAC metadata")

                metadata = {}

                # Extract geospatial fields
                geo_fields = [
                    "GEOSPATIAL_CRS",
                    "GEOSPATIAL_WIDTH",
                    "GEOSPATIAL_HEIGHT",
                    "GEOSPATIAL_COUNT",
                    "GEOSPATIAL_DTYPE",
                    "GEOSPATIAL_NODATA",
                    "GEOSPATIAL_DATA_MIN",
                    "GEOSPATIAL_DATA_MAX",
                    "GEOSPATIAL_TRANSFORM",
                    "GEOSPATIAL_BOUNDS",
                    "GEOSPATIAL_SPATIAL_TILING",
                ]

                for field in geo_fields:
                    if field in flac_file:
                        value = flac_file[field][0]
                        key = field.replace("GEOSPATIAL_", "").lower()

                        # Convert types
                        if key in ["width", "height", "count"]:
                            metadata[key] = int(value) if value else 0
                        elif key in ["data_min", "data_max"]:
                            metadata[key] = float(value) if value else 0.0
                        elif key in ["transform", "bounds"]:
                            metadata[key] = json.loads(value) if value else []
                        elif key == "spatial_tiling":
                            metadata[key] = value.lower() == "true"
                        elif key == "nodata":
                            metadata[key] = (
                                None if value == "None" else float(value) if value else None
                            )
                        else:
                            metadata[key] = value

                return metadata
            else:
                raise ValueError("No embedded metadata found")

        except (ImportError, Exception) as e:
            self.logger.warning(f"Failed to read embedded metadata: {e}")

            # Fallback to JSON sidecar
            metadata_path = flac_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    return json.load(f)

        return None

    def _get_write_callback(self, output_path: Path):
        """Create a write callback for FLAC encoder"""
        self.output_file = open(output_path, "wb")

        def callback(data, num_bytes, num_samples, current_frame):
            self.output_file.write(data[:num_bytes])
            return True

        return callback
