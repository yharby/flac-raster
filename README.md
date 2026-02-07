# FLAC-Raster: Experimental Raster to FLAC Converter

[![CI/CD](https://github.com/yharby/flac-raster/actions/workflows/ci.yml/badge.svg)](https://github.com/yharby/flac-raster/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/flac-raster.svg)](https://badge.fury.io/py/flac-raster)
[![Python versions](https://img.shields.io/pypi/pyversions/flac-raster.svg)](https://pypi.org/project/flac-raster/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An experimental CLI tool that converts TIFF raster data files into FLAC audio format while preserving all geospatial metadata, CRS, and bounds information. This proof-of-concept explores using FLAC's lossless compression for geospatial data storage and introduces **HTTP range streaming** for efficient geospatial data access - **"Netflix for Geospatial Data"**.

## Features

### Core Capabilities

- **Bidirectional conversion**: TIFF to FLAC and FLAC to TIFF
- **Complete metadata preservation**: CRS, bounds, transform, data type, nodata values
- **All satellite data types supported**: uint8, int8, uint16, int16, uint32, int32, float32, float64
- **Multi-band support**: Up to 8 bands (RGB, multispectral, hyperspectral)
- **Lossless compression**: 7-15x size reduction with perfect reconstruction
- **Embedded metadata**: All geospatial metadata stored directly in FLAC files (no sidecar files)

### Remote Access (NEW in v0.2.0)

- **HTTP/HTTPS URLs**: Direct access to remote FLAC files
- **Cloud Storage**: Native support for S3, Azure Blob, and Google Cloud Storage via obstore
- **Async Reading**: High-performance async COG reading via async-geotiff
- **HTTP Range Requests**: Stream only the tiles you need (99%+ bandwidth savings)

### Streaming Architecture

FLAC-Raster supports two formats:

1. **Standard Format** - Single FLAC file, highest compression ratio
2. **Streaming Format** - Netflix-style independent tiles for HTTP range streaming

## Installation

### Using Pixi (Recommended)

```bash
# Install pixi (cross-platform package manager)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and setup
git clone https://github.com/yharby/flac-raster.git
cd flac-raster

# Install with default features
pixi install

# Or with cloud storage support
pixi install -e cloud

# Or with async COG reading
pixi install -e async

# Or with all features
pixi install -e full
```

### Using pip

```bash
# Basic installation
pip install flac-raster

# With cloud storage support (S3, Azure, GCS)
pip install flac-raster[cloud]

# With async COG reading
pip install flac-raster[async]

# With all optional features
pip install flac-raster[all]
```

## Usage

### CLI Commands

FLAC-Raster provides 5 main commands:

#### 1. Convert

Convert between TIFF and FLAC formats:

```bash
# TIFF to FLAC
flac-raster convert input.tif -o output.flac

# FLAC to TIFF
flac-raster convert input.flac -o output.tif

# With spatial tiling (for streaming)
flac-raster convert input.tif --streaming --tile-size 1024 -o streaming.flac

# From remote URL
flac-raster convert https://example.com/data.tif -o output.flac
flac-raster convert s3://bucket/data.tif -o output.flac
```

Options:

- `--output, -o`: Output file path
- `--compression, -c`: FLAC compression level 0-8 (default: 5)
- `--streaming`: Create streaming format with independent tiles
- `--tile-size`: Tile size in pixels (default: 512)
- `--force, -f`: Overwrite existing files
- `--verbose, -v`: Enable verbose logging

#### 2. Info

Display file information:

```bash
# Local file
flac-raster info data.tif
flac-raster info data.flac

# Remote file
flac-raster info https://example.com/data.flac
flac-raster info s3://bucket/data.tif
```

#### 3. Extract

Extract tiles from streaming FLAC files:

```bash
# Extract by tile ID
flac-raster extract streaming.flac --tile-id 42 -o tile.tif

# Extract by bounding box
flac-raster extract streaming.flac --bbox "xmin,ymin,xmax,ymax" -o area.tif

# Extract center tile
flac-raster extract streaming.flac --center -o center.tif

# From remote URL (downloads only the requested tile)
flac-raster extract https://cdn.example.com/streaming.flac --center -o center.tif
```

Options:

- `--tile-id`: Extract specific tile by ID
- `--bbox, -b`: Bounding box as 'xmin,ymin,xmax,ymax'
- `--center`: Extract center tile
- `--last`: Extract last tile
- `--output, -o`: Output file path (required)

#### 4. Query

Query spatial index and find tiles:

```bash
# Find tiles intersecting a bounding box
flac-raster query spatial.flac --bbox "-105.3,40.3,-105.1,40.5"

# Get byte ranges for HTTP streaming
flac-raster query spatial.flac --bbox "xmin,ymin,xmax,ymax" --format ranges
```

Options:

- `--bbox, -b`: Bounding box to query (required)
- `--format, -f`: Output format: 'ranges' or 'data'
- `--output, -o`: Output file for extracted data

#### 5. Compare

Compare two TIFF files for verification:

```bash
flac-raster compare original.tif reconstructed.tif

# Export comparison to JSON
flac-raster compare original.tif reconstructed.tif --export comparison.json
```

### Python API

```python
from flac_raster import (
    RasterFLACConverter,
    SpatialFLACEncoder,
    SpatialFLACStreamer,
    normalize_to_audio,
    denormalize_from_audio,
)

# Basic conversion
converter = RasterFLACConverter()
converter.tiff_to_flac("input.tif", "output.flac")
converter.flac_to_tiff("output.flac", "reconstructed.tif")

# Spatial encoding with streaming support
encoder = SpatialFLACEncoder(tile_size=1024)
encoder.encode("input.tif", "streaming.flac", streaming=True)

# Stream tiles from remote
streamer = SpatialFLACStreamer("https://example.com/streaming.flac")
tile_data, metadata = streamer.get_tile_by_id(42)
tiles = streamer.get_tiles_by_bbox(xmin, ymin, xmax, ymax)
```

### Async API (Optional)

```python
from flac_raster import AsyncGeoTIFFReader, read_geotiff_async

# Async reading from cloud storage
async with AsyncGeoTIFFReader("s3://bucket/data.tif") as reader:
    # Read full data
    data = await reader.read()

    # Read a window
    window_data = await reader.read_window(0, 0, 512, 512)

    # Read a tile
    tile_data = await reader.read_tile(tile_x=2, tile_y=3)

# Functional API
data, metadata = await read_geotiff_async("https://example.com/cog.tif")
```

## Supported Data Types

| Data Type | Bits   | Use Cases                                   |
| --------- | ------ | ------------------------------------------- |
| uint8     | 8-bit  | RGB composites, classification maps         |
| int8      | 8-bit  | Signed byte data                            |
| uint16    | 16-bit | Sentinel-2, Landsat, most satellite imagery |
| int16     | 16-bit | DEMs, signed integer data                   |
| uint32    | 32-bit | Large count data                            |
| int32     | 32-bit | Signed 32-bit integer data                  |
| float32   | 32-bit | Reflectance, NDVI, processed data           |
| float64   | 64-bit | High-precision floating point               |

All data types are properly normalized to the audio range and can be perfectly reconstructed.

## Remote URL Support

FLAC-Raster supports multiple remote storage backends:

| Protocol     | Example                         | Requirements                     |
| ------------ | ------------------------------- | -------------------------------- |
| HTTP/HTTPS   | `https://example.com/data.flac` | Built-in                         |
| Amazon S3    | `s3://bucket/path/data.tif`     | `pip install flac-raster[cloud]` |
| Azure Blob   | `az://container/path/data.tif`  | `pip install flac-raster[cloud]` |
| Google Cloud | `gs://bucket/path/data.tif`     | `pip install flac-raster[cloud]` |

For cloud storage, credentials are read from environment variables or default credential chains.

## Performance

### Compression Results

| Dataset                          | Original | FLAC   | Compression |
| -------------------------------- | -------- | ------ | ----------- |
| DEM (1201x1201, int16)           | 2.8 MB   | 185 KB | 15.25x      |
| Multispectral (200x200x6, uint8) | 235 KB   | 32 KB  | 7.38x       |
| RGB (256x256x3, uint8)           | 193 KB   | 27 KB  | 7.26x       |

### HTTP Range Streaming Efficiency

| Use Case    | Download Size | Full File | Savings |
| ----------- | ------------- | --------- | ------- |
| Single tile | 1.5 MB        | 185 MB    | 99.2%   |
| Corner tile | 0.8 MB        | 185 MB    | 99.5%   |
| Bbox query  | 0.8-1.5 MB    | 185 MB    | 99%+    |

## Technical Details

### Data Flow

```
GeoTIFF Input
    |
    v
Read raster data + metadata
    |
    v
Normalize to audio range [-1, 1]
    |
    v
Convert to int16/int32 PCM
    |
    v
Encode as FLAC (multi-channel)
    |
    v
Embed metadata in VORBIS_COMMENT
    |
    v
FLAC Output
```

### Embedded Metadata

All geospatial information is stored in FLAC VORBIS_COMMENT blocks:

```
GEOSPATIAL_CRS=EPSG:4326
GEOSPATIAL_WIDTH=1201
GEOSPATIAL_HEIGHT=1201
GEOSPATIAL_TRANSFORM=...
GEOSPATIAL_BOUNDS=...
GEOSPATIAL_DATA_MIN=...
GEOSPATIAL_DATA_MAX=...
GEOSPATIAL_SPATIAL_INDEX=<base64(gzip(json))>
```

### Streaming Format Structure

```
[4 bytes: index size]
[JSON spatial index]
[Complete FLAC Tile 1]
[Complete FLAC Tile 2]
...
[Complete FLAC Tile N]
```

Each tile is a complete, self-contained FLAC file that can be decoded independently.

## Limitations

- Maximum 8 bands (FLAC channel limitation)
- Minimum 16-bit encoding (pyflac decoder limitation)
- FLAC bit depths: 16 or 24-bit only
- Large rasters may take time to process
- Experimental: Not recommended for production without thorough testing

## Project Structure

```
flac-raster/
├── src/flac_raster/
│   ├── __init__.py           # Package exports
│   ├── cli.py                # Command-line interface
│   ├── converter.py          # Core conversion logic
│   ├── spatial_encoder.py    # Spatial tiling and streaming
│   ├── normalization.py      # Data normalization (NEW)
│   ├── remote.py             # Remote file access (NEW)
│   ├── async_reader.py       # Async COG reading (NEW)
│   ├── metadata_encoder.py   # Embedded metadata handling
│   └── compare.py            # Comparison utilities
├── tests/                    # Test suite
├── docs/                     # Documentation
│   └── TECHNICAL_ANALYSIS.md # Technical analysis with diagrams
├── examples/                 # Example scripts
├── pixi.toml                 # Pixi configuration
├── pyproject.toml            # Python project configuration
└── README.md                 # This file
```

## Development

### Environment Setup (Pixi + uv)

Pixi provides Python and GDAL/rasterio (conda), uv manages Python packages:

```bash
# Install base environment (python, uv, rasterio)
pixi install

# Sync Python dependencies with uv
pixi run install          # or: pixi run uv sync
pixi run install-dev      # with all extras: pixi run uv sync --all-extras
```

### Running Commands

```bash
# Run commands via pixi tasks (which use uv run internally)
pixi run test             # pytest tests/
pixi run lint             # ruff check
pixi run format           # ruff format

# Or use uv directly
pixi run uv run flac-raster --help
pixi run uv run python -c "import flac_raster; print(flac_raster.__version__)"
```

### Adding Dependencies

```bash
# Add Python package (via uv)
pixi run uv add some-package

# Add dev dependency
pixi run uv add --dev pytest-xdist

# Add optional dependency group
pixi run uv add --optional cloud boto3

# Update all dependencies to latest
pixi run uv lock --upgrade
pixi run uv sync
```

### Building and Publishing

```bash
# Build the package
pixi run build            # or: pixi run uv build

# Publish to PyPI
pixi run uv publish

# Or publish to test PyPI first
pixi run uv publish --index testpypi
```

## Documentation

- [Sentinel-2 Tutorial](docs/SENTINEL2_TUTORIAL.md) - Step-by-step guide with real satellite data
- [Technical Analysis](docs/TECHNICAL_ANALYSIS.md) - Detailed technical analysis with Mermaid diagrams
- [Publishing Guide](PUBLISHING.md) - Instructions for publishing releases

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test them
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Create a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.
