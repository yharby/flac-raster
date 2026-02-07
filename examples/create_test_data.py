#!/usr/bin/env python3
"""
Create test data for FLAC-Raster examples
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def create_dem_sample(filename="sample_dem.tif", size=(512, 512)):
    """Create a sample DEM-like elevation dataset"""
    width, height = size

    # Create elevation-like pattern
    x = np.linspace(0, 20, width)
    y = np.linspace(0, 20, height)
    X, Y = np.meshgrid(x, y)

    # Create realistic elevation data with multiple peaks
    elevation = (
        1000  # Base elevation
        + 300 * np.sin(X * 0.5) * np.cos(Y * 0.3)  # Main terrain
        + 150 * np.sin(X * 1.2) * np.sin(Y * 1.1)  # Secondary features
        + 50 * np.random.rand(height, width)  # Noise
    ).astype(np.int16)

    # Define transform (place in Colorado, USA)
    transform = from_origin(-105.5, 40.5, 0.001, 0.001)

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=elevation.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(elevation, 1)

    print(f"Created DEM sample: {filename}")
    print(f"  Size: {width}x{height}")
    print(f"  Elevation range: {elevation.min()} - {elevation.max()} meters")
    return filename


def create_rgb_sample(filename="sample_rgb.tif", size=(256, 256)):
    """Create a sample RGB image"""
    width, height = size

    # Create different patterns for R, G, B
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(x, y)

    # Red channel: radial gradient
    r = (128 + 127 * np.cos(np.sqrt(X**2 + Y**2))).astype(np.uint8)

    # Green channel: horizontal waves
    g = (128 + 127 * np.sin(X)).astype(np.uint8)

    # Blue channel: vertical waves
    b = (128 + 127 * np.sin(Y)).astype(np.uint8)

    # Stack bands
    rgb_data = np.stack([r, g, b])

    # Define transform (place in California, USA)
    transform = from_origin(-120.0, 37.0, 0.0001, 0.0001)

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=rgb_data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(rgb_data)

    print(f"Created RGB sample: {filename}")
    print(f"  Size: {width}x{height}")
    print("  Bands: 3 (RGB)")
    return filename


def create_multispectral_sample(filename="sample_multispectral.tif", size=(200, 200)):
    """Create a sample multispectral image (6 bands)"""
    width, height = size
    bands = 6

    # Create different spectral signatures
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)

    data = []
    for i in range(bands):
        # Each band has different frequency and phase
        freq = 0.5 + i * 0.3
        phase = i * np.pi / 3
        band = (128 + 100 * np.sin(X * freq + phase) * np.cos(Y * freq)).astype(np.uint8)
        data.append(band)

    multispectral_data = np.stack(data)

    # Define transform
    transform = from_origin(-75.0, 35.0, 0.0001, 0.0001)

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=multispectral_data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(multispectral_data)

    print(f"Created multispectral sample: {filename}")
    print(f"  Size: {width}x{height}")
    print(f"  Bands: {bands}")
    return filename


if __name__ == "__main__":
    print("Creating test datasets for FLAC-Raster...")
    print("=" * 50)

    # Create test data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    # Create different types of test data
    dem_file = create_dem_sample(test_dir / "sample_dem.tif")
    rgb_file = create_rgb_sample(test_dir / "sample_rgb.tif")
    ms_file = create_multispectral_sample(test_dir / "sample_multispectral.tif")

    print("\n" + "=" * 50)
    print("Test data created! Try these commands:")
    print("=" * 50)

    print("\n1. Convert DEM to FLAC:")
    print(f"   flac-raster convert {dem_file} -o test_data/sample_dem.flac")

    print("\n2. Convert RGB to FLAC:")
    print(f"   flac-raster convert {rgb_file} -o test_data/sample_rgb.flac")

    print("\n3. Convert multispectral to FLAC:")
    print(f"   flac-raster convert {ms_file} -o test_data/sample_multispectral.flac")

    print("\n4. Get file info:")
    print(f"   flac-raster info {dem_file}")
    print("   flac-raster info test_data/sample_dem.flac")

    print("\n5. Convert back and compare:")
    print("   flac-raster convert test_data/sample_dem.flac -o test_data/dem_reconstructed.tif")
    print(f"   flac-raster compare {dem_file} test_data/dem_reconstructed.tif")

    print("\n6. Create streaming FLAC with tiles:")
    print(
        f"   flac-raster convert {dem_file} --streaming --tile-size 128 -o test_data/dem_streaming.flac"
    )

    print("\n7. Extract center tile from streaming FLAC:")
    print(
        "   flac-raster extract test_data/dem_streaming.flac --center -o test_data/center_tile.tif"
    )
