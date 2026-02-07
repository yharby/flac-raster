"""
Comparison utilities for FLAC-Raster
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger("flac_raster.compare")


def compare_tiffs(file1_path: Path, file2_path: Path, show_bands: bool = True) -> dict:
    """
    Compare two TIFF files and return comparison statistics

    Args:
        file1_path: Path to first TIFF file
        file2_path: Path to second TIFF file
        show_bands: Whether to show per-band statistics

    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing {file1_path} and {file2_path}")

    # Read both files
    with rasterio.open(file1_path) as src1:
        data1 = src1.read()
        meta1 = src1.meta.copy()

    with rasterio.open(file2_path) as src2:
        data2 = src2.read()
        meta2 = src2.meta.copy()

    # Basic comparison
    results = {
        "file1": file1_path.name,
        "file2": file2_path.name,
        "shape_match": data1.shape == data2.shape,
        "dtype_match": data1.dtype == data2.dtype,
        "crs_match": meta1.get("crs") == meta2.get("crs"),
        "file1_shape": data1.shape,
        "file2_shape": data2.shape,
        "file1_dtype": str(data1.dtype),
        "file2_dtype": str(data2.dtype),
        "file1_crs": str(meta1.get("crs", "None")),
        "file2_crs": str(meta2.get("crs", "None")),
    }

    # If shapes match, compute detailed statistics
    if results["shape_match"]:
        results["arrays_equal"] = np.array_equal(data1, data2)
        results["max_difference"] = float(np.max(np.abs(data1 - data2)))
        results["mean_difference"] = float(np.mean(np.abs(data1 - data2)))
        results["rmse"] = float(np.sqrt(np.mean((data1 - data2) ** 2)))

        # Overall data ranges
        results["file1_min"] = float(np.min(data1))
        results["file1_max"] = float(np.max(data1))
        results["file2_min"] = float(np.min(data2))
        results["file2_max"] = float(np.max(data2))

        # Per-band statistics
        if show_bands and data1.ndim == 3:
            results["bands"] = []
            for i in range(data1.shape[0]):
                band_stats = {
                    "band": i + 1,
                    "equal": np.array_equal(data1[i], data2[i]),
                    "max_diff": float(np.max(np.abs(data1[i] - data2[i]))),
                    "mean_diff": float(np.mean(np.abs(data1[i] - data2[i]))),
                    "file1_range": [float(data1[i].min()), float(data1[i].max())],
                    "file2_range": [float(data2[i].min()), float(data2[i].max())],
                }
                results["bands"].append(band_stats)

    return results


def display_comparison_table(results: dict):
    """Display comparison results in a nice table format"""

    # Create main comparison table
    table = Table(title="TIFF Comparison Results", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column(results["file1"], style="green")
    table.add_column(results["file2"], style="yellow")
    table.add_column("Match", style="bold")

    # Add basic properties
    table.add_row(
        "Shape",
        str(results["file1_shape"]),
        str(results["file2_shape"]),
        "YES" if results["shape_match"] else "NO",
    )
    table.add_row(
        "Data Type",
        results["file1_dtype"],
        results["file2_dtype"],
        "YES" if results["dtype_match"] else "NO",
    )
    table.add_row(
        "CRS", results["file1_crs"], results["file2_crs"], "YES" if results["crs_match"] else "NO"
    )

    console.print(table)

    # If shapes match, show detailed statistics
    if results.get("shape_match"):
        stats_table = Table(title="Statistical Comparison", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="bold")

        stats_table.add_row("Arrays Equal", "YES" if results["arrays_equal"] else "NO")
        stats_table.add_row("Max Difference", f"{results['max_difference']:.6f}")
        stats_table.add_row("Mean Difference", f"{results['mean_difference']:.6f}")
        stats_table.add_row("RMSE", f"{results['rmse']:.6f}")

        console.print(stats_table)

        # Data ranges table
        range_table = Table(title="Data Ranges", show_header=True)
        range_table.add_column("File", style="cyan")
        range_table.add_column("Min", style="blue")
        range_table.add_column("Max", style="red")

        range_table.add_row(
            results["file1"], f"{results['file1_min']:.2f}", f"{results['file1_max']:.2f}"
        )
        range_table.add_row(
            results["file2"], f"{results['file2_min']:.2f}", f"{results['file2_max']:.2f}"
        )

        console.print(range_table)

        # Per-band statistics if available
        if "bands" in results:
            band_table = Table(title="Per-Band Statistics", show_header=True)
            band_table.add_column("Band", style="cyan")
            band_table.add_column("Equal", style="bold")
            band_table.add_column("Max Diff", style="yellow")
            band_table.add_column("Mean Diff", style="yellow")
            band_table.add_column(f"{results['file1']} Range", style="green")
            band_table.add_column(f"{results['file2']} Range", style="blue")

            for band in results["bands"]:
                band_table.add_row(
                    str(band["band"]),
                    "YES" if band["equal"] else "NO",
                    f"{band['max_diff']:.3f}",
                    f"{band['mean_diff']:.6f}",
                    f"[{band['file1_range'][0]:.1f}, {band['file1_range'][1]:.1f}]",
                    f"[{band['file2_range'][0]:.1f}, {band['file2_range'][1]:.1f}]",
                )

            console.print(band_table)
    else:
        console.print("[red]Cannot compute detailed statistics - shapes don't match![/red]")
