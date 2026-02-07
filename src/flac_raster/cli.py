"""
Command-line interface for FLAC-Raster

Simple, intuitive CLI for converting between GeoTIFF and FLAC formats
with support for local files and remote URLs (HTTP, S3, Azure, GCS).
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .compare import compare_tiffs, display_comparison_table
from .converter import RasterFLACConverter
from .remote import download_remote, is_remote_url

app = typer.Typer(
    name="flac-raster",
    help="Convert GeoTIFF raster data to/from FLAC format with spatial streaming support.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("flac_raster")


def _resolve_input(input_path: str, temp_files: list) -> Path:
    """Resolve input path, downloading if remote."""
    if is_remote_url(input_path):
        console.print(f"[cyan]Downloading remote file: {input_path}[/cyan]")
        local_path = download_remote(input_path)
        temp_files.append(local_path)
        return local_path
    return Path(input_path)


@app.command()
def convert(
    input_file: str = typer.Argument(
        ...,
        help="Input file (TIFF or FLAC). Supports local paths and URLs (http://, s3://, az://, gs://)",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (auto-generated if not provided)"
    ),
    compression_level: int = typer.Option(
        5, "--compression", "-c", min=0, max=8, help="FLAC compression level (0=fastest, 8=best)"
    ),
    spatial: bool = typer.Option(
        False, "--spatial", "-s", help="Enable spatial tiling for streaming"
    ),
    tile_size: int = typer.Option(
        512, "--tile-size", "-t", help="Tile size in pixels (default: 512)"
    ),
    streaming: bool = typer.Option(
        False,
        "--streaming",
        help="Create Netflix-style streaming format (each tile is complete FLAC)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing output file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Convert between TIFF and FLAC formats.

    Examples:
        flac-raster convert input.tif -o output.flac
        flac-raster convert input.flac -o output.tif
        flac-raster convert input.tif --spatial --tile-size 256
        flac-raster convert s3://bucket/input.tif -o output.flac
    """
    if verbose:
        logging.getLogger("flac_raster").setLevel(logging.DEBUG)

    temp_files = []

    try:
        # Resolve input (download if remote)
        input_path = _resolve_input(input_file, temp_files)

        if not input_path.exists():
            console.print(f"[red]Error: Input file does not exist: {input_path}[/red]")
            raise typer.Exit(1)

        # Determine conversion direction
        input_suffix = input_path.suffix.lower()
        if input_suffix in [".tif", ".tiff"]:
            conversion_type = "tiff_to_flac"
            default_suffix = ".flac"
        elif input_suffix == ".flac":
            conversion_type = "flac_to_tiff"
            default_suffix = ".tif"
        else:
            console.print(f"[red]Error: Unsupported format: {input_suffix}[/red]")
            console.print("[yellow]Supported: .tif, .tiff, .flac[/yellow]")
            raise typer.Exit(1)

        # Set output path
        if output_file is None:
            if streaming:
                output_file = input_path.with_name(f"{input_path.stem}_streaming{default_suffix}")
            else:
                output_file = input_path.with_suffix(default_suffix)

        # Check existing output
        if output_file.exists() and not force:
            console.print(f"[red]Error: Output exists: {output_file}[/red]")
            console.print("[yellow]Use --force to overwrite[/yellow]")
            raise typer.Exit(1)

        # Handle streaming format specially
        if streaming and conversion_type == "tiff_to_flac":
            _create_streaming_flac(input_path, output_file, tile_size, compression_level)
            return

        # Regular conversion
        converter = RasterFLACConverter()

        if conversion_type == "tiff_to_flac":
            result = converter.tiff_to_flac(
                input_path, output_file, compression_level, spatial, tile_size
            )
            if spatial and result:
                console.print(f"[green]Created {len(result.frames)} spatial tiles[/green]")
        else:
            converter.flac_to_tiff(input_path, output_file)

    except Exception as e:
        logger.exception("Conversion failed")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        # Cleanup temp files
        for tmp in temp_files:
            if tmp.exists():
                tmp.unlink()


@app.command()
def info(
    file_path: str = typer.Argument(..., help="File to inspect (local or remote URL)"),
):
    """
    Display information about a FLAC or TIFF file.

    Examples:
        flac-raster info input.tif
        flac-raster info output.flac
        flac-raster info https://example.com/data.flac
        flac-raster info s3://bucket/data.tif
    """
    temp_files = []

    try:
        # Resolve input
        local_path = _resolve_input(file_path, temp_files)

        if not local_path.exists():
            console.print(f"[red]Error: File not found: {local_path}[/red]")
            raise typer.Exit(1)

        suffix = local_path.suffix.lower()

        if suffix in [".tif", ".tiff"]:
            _show_tiff_info(local_path)
        elif suffix == ".flac":
            _show_flac_info(local_path, is_remote=is_remote_url(file_path))
        else:
            console.print(f"[red]Unsupported format: {suffix}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        logger.exception("Info failed")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        for tmp in temp_files:
            if tmp.exists():
                tmp.unlink()


@app.command()
def extract(
    flac_file: str = typer.Argument(..., help="Streaming FLAC file (local or remote URL)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output TIFF file path"),
    bbox: Optional[str] = typer.Option(
        None, "--bbox", "-b", help="Bounding box: 'xmin,ymin,xmax,ymax'"
    ),
    tile_id: Optional[int] = typer.Option(None, "--tile-id", help="Extract specific tile by ID"),
    center: bool = typer.Option(False, "--center", help="Extract center tile"),
    last: bool = typer.Option(False, "--last", help="Extract last tile"),
):
    """
    Extract tiles from a streaming FLAC file.

    Supports Netflix-style streaming with minimal bandwidth usage.

    Examples:
        flac-raster extract data.flac -o tile.tif --center
        flac-raster extract data.flac -o tile.tif --tile-id 42
        flac-raster extract data.flac -o tile.tif --bbox "100,200,300,400"
        flac-raster extract s3://bucket/data.flac -o tile.tif --last
    """
    import struct

    try:
        # Load streaming metadata
        console.print(f"[cyan]Loading streaming metadata from: {flac_file}[/cyan]")

        if is_remote_url(flac_file):
            from .remote import RemoteFile

            remote = RemoteFile(flac_file)
            index_size_bytes = remote.read_range(0, 3)
            index_size = struct.unpack(">I", index_size_bytes)[0]
            index_json = remote.read_range(4, 3 + index_size)
            metadata = json.loads(index_json.decode("utf-8"))
        else:
            with open(flac_file, "rb") as f:
                index_size = struct.unpack(">I", f.read(4))[0]
                metadata = json.loads(f.read(index_size).decode("utf-8"))

        frames = metadata["frames"]
        console.print(f"[green]Found {len(frames)} tiles[/green]")

        # Determine target frame
        target_frame = None

        if tile_id is not None:
            target_frame = next((f for f in frames if f["frame_id"] == tile_id), None)
            if not target_frame:
                console.print(f"[red]Tile ID {tile_id} not found[/red]")
                raise typer.Exit(1)
        elif last:
            target_frame = max(frames, key=lambda f: f["frame_id"])
        elif center:
            # Find center tile
            all_bboxes = [f["bbox"] for f in frames]
            center_x = (min(b[0] for b in all_bboxes) + max(b[2] for b in all_bboxes)) / 2
            center_y = (min(b[1] for b in all_bboxes) + max(b[3] for b in all_bboxes)) / 2

            target_frame = min(
                frames,
                key=lambda f: (
                    ((f["bbox"][0] + f["bbox"][2]) / 2 - center_x) ** 2
                    + ((f["bbox"][1] + f["bbox"][3]) / 2 - center_y) ** 2
                ),
            )
        elif bbox:
            bbox_coords = [float(x.strip()) for x in bbox.split(",")]
            if len(bbox_coords) != 4:
                console.print("[red]Bbox must have 4 coordinates[/red]")
                raise typer.Exit(1)

            # Find intersecting tiles
            intersecting = [
                f
                for f in frames
                if (
                    bbox_coords[0] < f["bbox"][2]
                    and bbox_coords[2] > f["bbox"][0]
                    and bbox_coords[1] < f["bbox"][3]
                    and bbox_coords[3] > f["bbox"][1]
                )
            ]
            if not intersecting:
                console.print("[red]No tiles intersect bbox[/red]")
                raise typer.Exit(1)
            target_frame = intersecting[0]
            if len(intersecting) > 1:
                console.print(
                    f"[yellow]Using first of {len(intersecting)} intersecting tiles[/yellow]"
                )
        else:
            console.print("[red]Specify --tile-id, --bbox, --center, or --last[/red]")
            raise typer.Exit(1)

        # Download and convert tile
        console.print(f"[cyan]Extracting tile {target_frame['frame_id']}[/cyan]")
        console.print(f"  Bbox: {target_frame['bbox']}")
        console.print(f"  Size: {target_frame['byte_size']:,} bytes")

        header_size = 4 + index_size
        abs_start = header_size + target_frame["byte_offset"]
        abs_end = abs_start + target_frame["byte_size"] - 1

        if is_remote_url(flac_file):
            tile_data = remote.read_range(abs_start, abs_end)
        else:
            with open(flac_file, "rb") as f:
                f.seek(abs_start)
                tile_data = f.read(target_frame["byte_size"])

        # Convert tile to TIFF
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            tmp.write(tile_data)
            tmp_path = Path(tmp.name)

        try:
            converter = RasterFLACConverter()
            converter.flac_to_tiff(tmp_path, output)

            # Show bandwidth savings
            total_bytes = sum(f["byte_size"] for f in frames)
            savings = (1 - target_frame["byte_size"] / total_bytes) * 100
            console.print(f"[green]Saved to: {output}[/green]")
            console.print(
                f"[blue]Bandwidth: {target_frame['byte_size'] / 1024:.1f} KB "
                f"(saved {savings:.1f}%)[/blue]"
            )
        finally:
            tmp_path.unlink()

    except Exception as e:
        logger.exception("Extraction failed")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    flac_file: str = typer.Argument(..., help="Spatial FLAC file (local or remote)"),
    bbox: str = typer.Option(..., "--bbox", "-b", help="Bounding box: 'xmin,ymin,xmax,ymax'"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save byte ranges to JSON file"
    ),
):
    """
    Query spatial FLAC file by bounding box.

    Returns HTTP byte ranges for efficient partial downloads.

    Examples:
        flac-raster query data.flac --bbox "34.1,28.6,34.3,28.8"
    """
    try:
        from .spatial_encoder import SpatialFLACStreamer

        # Parse bbox
        bbox_coords = tuple(float(x.strip()) for x in bbox.split(","))
        if len(bbox_coords) != 4:
            console.print("[red]Bbox must have 4 coordinates[/red]")
            raise typer.Exit(1)

        # Load spatial index
        console.print("[cyan]Loading spatial index...[/cyan]")
        streamer = SpatialFLACStreamer(flac_file)

        # Query
        ranges = streamer.get_byte_ranges_for_bbox(bbox_coords)
        total_bytes = sum(end - start + 1 for start, end in ranges)

        # Display results
        table = Table(title=f"Byte Ranges for bbox {bbox}")
        table.add_column("#", style="cyan")
        table.add_column("Start", style="green")
        table.add_column("End", style="yellow")
        table.add_column("Size", style="blue")
        table.add_column("Range Header", style="magenta")

        for i, (start, end) in enumerate(ranges, 1):
            table.add_row(
                str(i),
                f"{start:,}",
                f"{end:,}",
                f"{end - start + 1:,}",
                f"bytes={start}-{end}",
            )

        console.print(table)
        console.print(f"[bold]Total: {total_bytes:,} bytes ({len(ranges)} ranges)[/bold]")

        # Save to file if requested
        if output:
            data = {
                "bbox": list(bbox_coords),
                "ranges": [{"start": s, "end": e} for s, e in ranges],
                "total_bytes": total_bytes,
            }
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
            console.print(f"[green]Saved to: {output}[/green]")

    except Exception as e:
        logger.exception("Query failed")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    file1: Path = typer.Argument(..., help="First TIFF file"),
    file2: Path = typer.Argument(..., help="Second TIFF file"),
    show_bands: bool = typer.Option(
        True, "--show-bands/--no-bands", help="Show per-band statistics"
    ),
    export_json: Optional[Path] = typer.Option(
        None, "--export", "-e", help="Export comparison to JSON"
    ),
):
    """
    Compare two TIFF files and display statistics.

    Useful for verifying lossless round-trip conversion.

    Examples:
        flac-raster compare original.tif reconstructed.tif
        flac-raster compare original.tif reconstructed.tif --export results.json
    """
    for f in [file1, file2]:
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)
        if f.suffix.lower() not in [".tif", ".tiff"]:
            console.print(f"[red]Not a TIFF file: {f}[/red]")
            raise typer.Exit(1)

    try:
        results = compare_tiffs(file1, file2, show_bands)
        display_comparison_table(results)

        if export_json:
            with open(export_json, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Exported to: {export_json}[/green]")

    except Exception as e:
        logger.exception("Comparison failed")
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _show_tiff_info(path: Path):
    """Display TIFF file information."""
    import rasterio

    with rasterio.open(path) as src:
        table = Table(title=f"TIFF: {path.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Dimensions", f"{src.width} x {src.height}")
        table.add_row("Bands", str(src.count))
        table.add_row("Data Type", str(src.dtypes[0]))
        table.add_row("CRS", str(src.crs))
        table.add_row(
            "Bounds",
            f"({src.bounds.left:.6f}, {src.bounds.bottom:.6f}, {src.bounds.right:.6f}, {src.bounds.top:.6f})",
        )
        table.add_row("File Size", f"{path.stat().st_size / 1024 / 1024:.2f} MB")

        console.print(table)


def _show_flac_info(path: Path, is_remote: bool = False):
    """Display FLAC file information."""
    import pyflac

    table = Table(title=f"FLAC: {path.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Basic FLAC info
    try:
        decoder = pyflac.FileDecoder(str(path))
        audio_data, sample_rate = decoder.process()
        table.add_row("Sample Rate", f"{sample_rate} Hz")
        table.add_row("Channels", str(audio_data.shape[1] if len(audio_data.shape) > 1 else 1))
        table.add_row("Audio Shape", str(audio_data.shape))
        table.add_row("Audio Type", str(audio_data.dtype))
    except Exception:
        pass

    table.add_row("File Size", f"{path.stat().st_size / 1024 / 1024:.2f} MB")
    console.print(table)

    # Try to read geospatial metadata
    try:
        from mutagen.flac import FLAC

        flac_file = FLAC(str(path))
        if "GEOSPATIAL_CRS" in flac_file:
            geo_table = Table(title="Geospatial Metadata")
            geo_table.add_column("Property", style="cyan")
            geo_table.add_column("Value", style="green")

            geo_table.add_row(
                "Dimensions",
                f"{flac_file.get('GEOSPATIAL_WIDTH', ['?'])[0]} x {flac_file.get('GEOSPATIAL_HEIGHT', ['?'])[0]}",
            )
            geo_table.add_row("Bands", flac_file.get("GEOSPATIAL_COUNT", ["?"])[0])
            geo_table.add_row("Original Type", flac_file.get("GEOSPATIAL_DTYPE", ["?"])[0])
            geo_table.add_row("CRS", flac_file.get("GEOSPATIAL_CRS", ["?"])[0])
            geo_table.add_row(
                "Data Range",
                f"[{flac_file.get('GEOSPATIAL_DATA_MIN', ['?'])[0]}, {flac_file.get('GEOSPATIAL_DATA_MAX', ['?'])[0]}]",
            )
            geo_table.add_row(
                "Spatial Tiling", flac_file.get("GEOSPATIAL_SPATIAL_TILING", ["false"])[0]
            )

            console.print(geo_table)
    except Exception:
        pass


def _create_streaming_flac(
    input_path: Path, output_path: Path, tile_size: int, compression_level: int
):
    """Create Netflix-style streaming FLAC with self-contained tiles."""
    import rasterio
    from rasterio.windows import Window

    console.print("[cyan]Creating streaming FLAC...[/cyan]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}")
    console.print(f"  Tile size: {tile_size}x{tile_size}")

    converter = RasterFLACConverter()

    with rasterio.open(input_path) as src:
        console.print(f"  Dimensions: {src.width}x{src.height}")

        spatial_index = {
            "crs": str(src.crs),
            "transform": list(src.transform),
            "width": src.width,
            "height": src.height,
            "tile_size": tile_size,
            "frames": [],
        }

        tile_data_chunks = []
        total_offset = 0
        frame_id = 0

        for row_start in range(0, src.height, tile_size):
            for col_start in range(0, src.width, tile_size):
                tile_width = min(tile_size, src.width - col_start)
                tile_height = min(tile_size, src.height - row_start)

                window = Window(col_start, row_start, tile_width, tile_height)
                tile_data = src.read(window=window)

                tile_transform = src.window_transform(window)
                xmin = tile_transform.c
                ymax = tile_transform.f
                xmax = xmin + tile_width * tile_transform.a
                ymin = ymax + tile_height * tile_transform.e

                console.print(f"  Processing tile {frame_id}: {tile_width}x{tile_height}")

                # Create temporary files for conversion
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
                    tmp_tif_path = Path(tmp_tif.name)
                with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp_flac:
                    tmp_flac_path = Path(tmp_flac.name)

                try:
                    # Write tile as TIFF
                    with rasterio.open(
                        tmp_tif_path,
                        "w",
                        driver="GTiff",
                        height=tile_height,
                        width=tile_width,
                        count=tile_data.shape[0] if tile_data.ndim == 3 else 1,
                        dtype=tile_data.dtype,
                        crs=src.crs,
                        transform=tile_transform,
                    ) as dst:
                        if tile_data.ndim == 3:
                            dst.write(tile_data)
                        else:
                            dst.write(tile_data, 1)

                    # Convert to FLAC
                    converter.tiff_to_flac(tmp_tif_path, tmp_flac_path, compression_level)

                    # Read FLAC bytes
                    with open(tmp_flac_path, "rb") as f:
                        flac_bytes = f.read()

                finally:
                    tmp_tif_path.unlink()
                    tmp_flac_path.unlink()

                # Add to index
                spatial_index["frames"].append(
                    {
                        "frame_id": frame_id,
                        "bbox": [xmin, ymin, xmax, ymax],
                        "window": {
                            "col_off": col_start,
                            "row_off": row_start,
                            "width": tile_width,
                            "height": tile_height,
                        },
                        "byte_offset": total_offset,
                        "byte_size": len(flac_bytes),
                    }
                )

                tile_data_chunks.append(flac_bytes)
                total_offset += len(flac_bytes)
                frame_id += 1

    # Write streaming file
    with open(output_path, "wb") as f:
        index_json = json.dumps(spatial_index, separators=(",", ":")).encode("utf-8")
        f.write(len(index_json).to_bytes(4, "big"))
        f.write(index_json)
        for chunk in tile_data_chunks:
            f.write(chunk)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    avg_tile_kb = (total_offset / len(tile_data_chunks)) / 1024

    console.print("\n[green]Created streaming FLAC![/green]")
    console.print(f"  File: {output_path}")
    console.print(f"  Size: {file_size_mb:.2f} MB")
    console.print(f"  Tiles: {len(tile_data_chunks)}")
    console.print(f"  Avg tile: {avg_tile_kb:.1f} KB")


if __name__ == "__main__":
    app()
