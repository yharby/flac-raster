"""
Enhanced FLAC encoder with embedded metadata support
Uses VORBIS_COMMENT blocks to store geospatial metadata directly in FLAC files
"""

import base64
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rich.console import Console

console = Console()


class MetadataFLACEncoder:
    """FLAC encoder that embeds geospatial metadata using VORBIS_COMMENT blocks"""

    def __init__(self):
        self.logger = logging.getLogger("flac_raster.metadata_encoder")

    def _create_vorbis_comments(self, metadata: Dict) -> Dict[str, str]:
        """Create VORBIS_COMMENT fields from geospatial metadata"""

        # FLAC VORBIS_COMMENT standard fields + custom geospatial fields
        comments = {
            # Standard fields
            "TITLE": "Geospatial Raster Data",
            "DESCRIPTION": "TIFF raster converted to FLAC with spatial metadata",
            "ENCODER": "FLAC-Raster spatial encoder",
            # Custom geospatial fields (following convention: GEOSPATIAL_*)
            "GEOSPATIAL_CRS": str(metadata.get("crs", "")),
            "GEOSPATIAL_WIDTH": str(metadata.get("width", 0)),
            "GEOSPATIAL_HEIGHT": str(metadata.get("height", 0)),
            "GEOSPATIAL_COUNT": str(metadata.get("count", 1)),
            "GEOSPATIAL_DTYPE": str(metadata.get("dtype", "")),
            "GEOSPATIAL_NODATA": str(metadata.get("nodata", "")),
            # Transform as base64 encoded JSON (VORBIS_COMMENT fields must be strings)
            "GEOSPATIAL_TRANSFORM": base64.b64encode(
                json.dumps(metadata.get("transform", [])).encode()
            ).decode("ascii"),
            # Bounds
            "GEOSPATIAL_BOUNDS": json.dumps(metadata.get("bounds", [])),
            # Data statistics
            "GEOSPATIAL_DATA_MIN": str(metadata.get("data_min", "")),
            "GEOSPATIAL_DATA_MAX": str(metadata.get("data_max", "")),
            # Spatial tiling info (if applicable)
            "GEOSPATIAL_SPATIAL_TILING": str(metadata.get("spatial_tiling", False)),
            "GEOSPATIAL_TILE_SIZE": str(metadata.get("tile_size", "")),
            "GEOSPATIAL_NUM_TILES": str(metadata.get("num_tiles", "")),
        }

        # Add spatial index as compressed JSON if available
        if "spatial_index" in metadata:
            spatial_json = json.dumps(metadata["spatial_index"], separators=(",", ":"))
            # Compress and encode spatial index
            import gzip

            compressed = gzip.compress(spatial_json.encode())
            comments["GEOSPATIAL_SPATIAL_INDEX"] = base64.b64encode(compressed).decode("ascii")

        return comments

    def _extract_metadata_from_vorbis(self, comments: Dict[str, str]) -> Dict:
        """Extract geospatial metadata from VORBIS_COMMENT fields"""

        metadata = {}

        # Extract standard geospatial fields
        geo_fields = {
            "crs": "GEOSPATIAL_CRS",
            "width": "GEOSPATIAL_WIDTH",
            "height": "GEOSPATIAL_HEIGHT",
            "count": "GEOSPATIAL_COUNT",
            "dtype": "GEOSPATIAL_DTYPE",
            "nodata": "GEOSPATIAL_NODATA",
            "data_min": "GEOSPATIAL_DATA_MIN",
            "data_max": "GEOSPATIAL_DATA_MAX",
            "spatial_tiling": "GEOSPATIAL_SPATIAL_TILING",
            "tile_size": "GEOSPATIAL_TILE_SIZE",
            "num_tiles": "GEOSPATIAL_NUM_TILES",
        }

        for key, comment_key in geo_fields.items():
            if comment_key in comments:
                value = comments[comment_key]
                # Convert to appropriate type
                if key in ["width", "height", "count", "tile_size", "num_tiles"]:
                    metadata[key] = int(value) if value else 0
                elif key in ["data_min", "data_max"]:
                    metadata[key] = float(value) if value else 0.0
                elif key == "spatial_tiling":
                    metadata[key] = value.lower() == "true"
                else:
                    metadata[key] = value

        # Decode transform
        if "GEOSPATIAL_TRANSFORM" in comments:
            try:
                transform_json = base64.b64decode(comments["GEOSPATIAL_TRANSFORM"]).decode()
                metadata["transform"] = json.loads(transform_json)
            except Exception as e:
                self.logger.warning(f"Failed to decode transform: {e}")

        # Decode bounds
        if "GEOSPATIAL_BOUNDS" in comments:
            try:
                metadata["bounds"] = json.loads(comments["GEOSPATIAL_BOUNDS"])
            except Exception as e:
                self.logger.warning(f"Failed to decode bounds: {e}")

        # Decode spatial index if available
        if "GEOSPATIAL_SPATIAL_INDEX" in comments:
            try:
                import gzip

                compressed = base64.b64decode(comments["GEOSPATIAL_SPATIAL_INDEX"])
                spatial_json = gzip.decompress(compressed).decode()
                metadata["spatial_index"] = json.loads(spatial_json)
            except Exception as e:
                self.logger.warning(f"Failed to decode spatial index: {e}")

        return metadata

    def encode_with_metadata(
        self,
        tiff_path: Path,
        flac_path: Path,
        compression_level: int = 5,
        spatial_tiling: bool = False,
        tile_size: int = 512,
    ) -> Optional[Dict]:
        """
        Encode TIFF to FLAC with embedded metadata using VORBIS_COMMENT

        Note: This is a simplified version. Full implementation would require
        custom FLAC encoding with APPLICATION metadata blocks for per-tile bbox data.
        For now, we store overall metadata in VORBIS_COMMENT and spatial index separately.
        """

        self.logger.info(
            f"Encoding TIFF to FLAC with embedded metadata: {tiff_path} -> {flac_path}"
        )

        # Read raster metadata
        with rasterio.open(tiff_path) as src:
            raster_data = src.read()

            # Prepare complete metadata
            metadata = {
                "crs": str(src.crs),
                "transform": list(src.transform),
                "bounds": list(src.bounds),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(raster_data.dtype),
                "nodata": src.nodata,
                "data_min": float(np.min(raster_data)),
                "data_max": float(np.max(raster_data)),
                "spatial_tiling": spatial_tiling,
                "tile_size": tile_size if spatial_tiling else None,
            }

        # For spatial tiling, use existing spatial encoder but with metadata embedding
        if spatial_tiling:
            self.logger.info("Using spatial tiling - will embed spatial index in metadata")
            from .spatial_encoder import SpatialFLACEncoder

            # Create spatial FLAC
            spatial_encoder = SpatialFLACEncoder(tile_size=tile_size)
            spatial_index = spatial_encoder.encode_spatial_flac(
                tiff_path, flac_path, compression_level
            )

            # Add spatial index to metadata
            metadata["spatial_index"] = spatial_index.to_dict()
            metadata["num_tiles"] = len(spatial_index.frames)

            # Now we need to add the metadata to the FLAC file
            # This would require rewriting the FLAC file with embedded metadata
            # For now, we'll use the hybrid approach: VORBIS comments + JSON sidecar
            self._embed_metadata_in_flac(flac_path, metadata)

            return spatial_index

        else:
            # Regular conversion with embedded metadata
            self.logger.info("Regular conversion with embedded metadata")

            # Use regular converter but modify to embed metadata
            from .converter import RasterFLACConverter

            converter = RasterFLACConverter()
            converter.tiff_to_flac(tiff_path, flac_path, compression_level)

            # Embed metadata in the FLAC file
            self._embed_metadata_in_flac(flac_path, metadata)

            return None

    def _embed_metadata_in_flac(self, flac_path: Path, metadata: Dict):
        """
        Embed metadata into FLAC file using available tools

        Note: This is a placeholder for proper FLAC metadata embedding.
        Current pyflac limitations mean we need alternative approaches.
        """

        # Create VORBIS_COMMENT data
        vorbis_comments = self._create_vorbis_comments(metadata)

        self.logger.info(f"Preparing to embed {len(vorbis_comments)} metadata fields")

        # For now, save as enhanced JSON sidecar with metadata embedding capability
        enhanced_metadata = {
            "format_version": "1.1",
            "embedded_in_flac": False,  # Will be True when we implement proper embedding
            "vorbis_comments": vorbis_comments,
            "metadata": metadata,
        }

        metadata_path = flac_path.with_suffix(".metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(enhanced_metadata, f, indent=2)

        self.logger.info(f"Enhanced metadata saved: {metadata_path}")

        # TODO: Implement actual FLAC metadata embedding
        # This would require:
        # 1. Reading existing FLAC file
        # 2. Adding VORBIS_COMMENT metadata block
        # 3. Rewriting FLAC file with metadata
        # 4. For spatial data: Adding APPLICATION blocks for tile metadata

        console.print(
            "[yellow]Note: Metadata saved as enhanced sidecar file. "
            "FLAC embedding requires libFLAC integration.[/yellow]"
        )

    def read_embedded_metadata(self, flac_path: Path) -> Optional[Dict]:
        """Read embedded metadata from FLAC file"""

        # First try to read enhanced metadata sidecar
        metadata_path = flac_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                enhanced_metadata = json.load(f)
                return enhanced_metadata.get("metadata", {})

        # Fallback to legacy JSON sidecar
        legacy_path = flac_path.with_suffix(".json")
        if legacy_path.exists():
            with open(legacy_path, "r") as f:
                return json.load(f)

        # TODO: Read from actual FLAC metadata when implemented
        self.logger.warning(f"No metadata found for {flac_path}")
        return None


class FLACMetadataReader:
    """Read metadata from FLAC files with embedded geospatial information"""

    def __init__(self):
        self.logger = logging.getLogger("flac_raster.metadata_reader")

    def extract_geospatial_info(self, flac_path: Path) -> Dict:
        """Extract all geospatial information from FLAC file"""

        # Try enhanced metadata first
        encoder = MetadataFLACEncoder()
        metadata = encoder.read_embedded_metadata(flac_path)

        if metadata:
            return {
                "has_embedded_metadata": True,
                "metadata": metadata,
                "spatial_tiling": metadata.get("spatial_tiling", False),
                "file_size": flac_path.stat().st_size,
            }
        else:
            return {
                "has_embedded_metadata": False,
                "metadata": {},
                "spatial_tiling": False,
                "file_size": flac_path.stat().st_size,
            }
