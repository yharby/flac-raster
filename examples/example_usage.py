#!/usr/bin/env python3
"""
Example usage of the FLAC-Raster converter

This script creates a sample TIFF file and shows CLI commands
for converting, inspecting, and comparing FLAC-Raster files.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin


def create_sample_tiff(filename="sample.tif"):
    """Create a sample TIFF file with geographic data"""
    # Create sample data (elevation-like pattern)
    width, height = 100, 100
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)

    # Create elevation-like data
    Z = 1000 + 100 * np.sin(X) * np.cos(Y) + 50 * np.random.rand(height, width)

    # Define transform (georeferencing)
    # Place at arbitrary coordinates
    transform = from_origin(-120.0, 40.0, 0.01, 0.01)

    # Write TIFF
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=Z.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(Z, 1)

    print(f"Created sample TIFF: {filename}")
    print(f"  Shape: {Z.shape}")
    print(f"  Data range: {Z.min():.2f} - {Z.max():.2f}")
    return filename


if __name__ == "__main__":
    # Create sample data
    tiff_file = create_sample_tiff()

    print("\n" + "=" * 50)
    print("CLI Commands:")
    print("=" * 50)

    print("\n1. Convert TIFF to FLAC:")
    print(f"   flac-raster convert {tiff_file} -o sample.flac")

    print("\n2. Convert FLAC back to TIFF:")
    print("   flac-raster convert sample.flac -o reconstructed.tif")

    print("\n3. Get file information:")
    print(f"   flac-raster info {tiff_file}")
    print("   flac-raster info sample.flac")

    print("\n4. Compare original and reconstructed:")
    print(f"   flac-raster compare {tiff_file} reconstructed.tif")

    print("\n5. Create streaming FLAC with spatial tiles:")
    print(f"   flac-raster convert {tiff_file} --streaming --tile-size 256 -o streaming.flac")

    print("\n6. Extract a tile from streaming FLAC:")
    print("   flac-raster extract streaming.flac --center -o center_tile.tif")

    print("\nYou can then open 'reconstructed.tif' in QGIS to verify the conversion!")
