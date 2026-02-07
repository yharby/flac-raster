# FLAC-Raster Tutorial: Working with Sentinel-2 Data

This tutorial demonstrates how to use FLAC-Raster with real Sentinel-2 satellite imagery, showing lossless compression and streaming capabilities.

## Test Data

We use a Sentinel-2 L2A tile from the AWS Earth Search STAC catalog:

- **Scene**: S2B_36RWV_20260201_0_L2A
- **Band**: B04 (Red, 10m resolution)
- **Location**: UTM Zone 36N (Middle East region)
- **Dimensions**: 10980 x 10980 pixels
- **Data Type**: uint16
- **Cloud Cover**: 0.08%

## Size Comparison

| Format | Size | Notes |
|--------|------|-------|
| Raw uncompressed | 230 MB | 10980 x 10980 x 2 bytes |
| Original COG (deflate) | 130 MB | Cloud-Optimized GeoTIFF |
| FLAC | 116 MB | 50% of raw, lossless |
| Streaming FLAC | 152 MB | 484 tiles for HTTP range requests |

## Step 1: Convert Remote Sentinel-2 to FLAC

Download and convert directly from AWS:

```bash
# Get the Sentinel-2 B04 band URL from STAC
B04_URL="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/36/R/WV/2026/2/S2B_36RWV_20260201_0_L2A/B04.tif"

# Convert remote TIFF to FLAC
flac-raster convert "$B04_URL" -o sentinel2_b04.flac
```

Output:
```
SUCCESS: Converted to FLAC: sentinel2_b04.flac
File size: 115.68 MB (compression: 11.3%)
```

## Step 2: Inspect FLAC File

```bash
flac-raster info sentinel2_b04.flac
```

Output:
```
                FLAC: sentinel2_b04.flac
+----------------+------------------+
| Property       | Value            |
+----------------+------------------+
| Sample Rate    | 192000 Hz        |
| Channels       | 1                |
| Audio Shape    | (120560400, 1)   |
+----------------+------------------+

              Geospatial Metadata
+----------------+------------------------------+
| Property       | Value                        |
+----------------+------------------------------+
| Dimensions     | 10980 x 10980                |
| Bands          | 1                            |
| Original Type  | uint16                       |
| CRS            | EPSG:32636 (UTM zone 36N)    |
| Data Range     | [0.0, 11672.0]               |
+----------------+------------------------------+
```

## Step 3: Convert FLAC Back to TIFF

```bash
flac-raster convert sentinel2_b04.flac -o sentinel2_reconstructed.tif
```

Output:
```
SUCCESS: Converted to TIFF: sentinel2_reconstructed.tif
```

## Step 4: Verify Lossless Round-Trip

```bash
flac-raster compare sentinel2_original.tif sentinel2_reconstructed.tif
```

Output:
```
              Statistical Comparison
+------------------+----------+
| Metric           | Value    |
+------------------+----------+
| Arrays Equal     | YES      |
| Max Difference   | 0.000000 |
| Mean Difference  | 0.000000 |
| RMSE             | 0.000000 |
+------------------+----------+
```

## Step 5: Create Streaming FLAC (Netflix-Style Tiles)

For HTTP range request streaming, create a tiled FLAC:

```bash
flac-raster convert sentinel2_original.tif --streaming --tile-size 512 -o sentinel2_streaming.flac
```

Output:
```
Created streaming FLAC!
  File: sentinel2_streaming.flac
  Size: 151.58 MB
  Tiles: 484
  Avg tile: 320.5 KB
```

## Step 6: Extract Single Tile (99.7% Bandwidth Savings)

Extract only the center tile without downloading the full file:

```bash
flac-raster extract sentinel2_streaming.flac --center -o center_tile.tif
```

Output:
```
SUCCESS: Converted to TIFF: center_tile.tif
Saved to: center_tile.tif
Bandwidth: 452.7 KB (saved 99.7%)
```

## Step 7: Extract by Tile ID

```bash
flac-raster extract sentinel2_streaming.flac --tile-id 0 -o tile_0.tif
```

Output:
```
SUCCESS: Converted to TIFF: tile_0.tif
Saved to: tile_0.tif
Bandwidth: 460.8 KB (saved 99.7%)
```

## Step 8: Validate with GDAL

### Original Sentinel-2

```bash
gdalinfo sentinel2_original.tif
```

```
Driver: GTiff/GeoTIFF
Size is 10980, 10980
Coordinate System is:
PROJCRS["WGS 84 / UTM zone 36N", ...]
Origin = (499980.000000000000000,3500040.000000000000000)
Pixel Size = (10.000000000000000,-10.000000000000000)
Corner Coordinates:
Upper Left  (  499980.000, 3500040.000) ( 32d59'59.24"E, 31d38' 7.97"N)
Lower Left  (  499980.000, 3390240.000) ( 32d59'59.25"E, 30d38'41.35"N)
Upper Right (  609780.000, 3500040.000) ( 34d 9'27.31"E, 31d37'49.08"N)
Lower Right (  609780.000, 3390240.000) ( 34d 8'44.22"E, 30d38'23.18"N)
Center      (  554880.000, 3445140.000) ( 33d34'32.50"E, 31d 8'20.10"N)
Band 1 Block=1024x1024 Type=UInt16, ColorInterp=Gray
  NoData Value=0
```

### Reconstructed from FLAC

```bash
gdalinfo sentinel2_reconstructed.tif
```

```
Driver: GTiff/GeoTIFF
Size is 10980, 10980
Coordinate System is:
PROJCRS["WGS 84 / UTM zone 36N", ...]
Origin = (499980.000000000000000,3500040.000000000000000)
Pixel Size = (10.000000000000000,-10.000000000000000)
Corner Coordinates:
Upper Left  (  499980.000, 3500040.000) ( 32d59'59.24"E, 31d38' 7.97"N)
Lower Left  (  499980.000, 3390240.000) ( 32d59'59.25"E, 30d38'41.35"N)
Upper Right (  609780.000, 3500040.000) ( 34d 9'27.31"E, 31d37'49.08"N)
Lower Right (  609780.000, 3390240.000) ( 34d 8'44.22"E, 30d38'23.18"N)
Center      (  554880.000, 3445140.000) ( 33d34'32.50"E, 31d 8'20.10"N)
Band 1 Block=10980x1 Type=UInt16, ColorInterp=Gray
  NoData Value=0
```

### Extracted Center Tile

```bash
gdalinfo center_tile.tif
```

```
Driver: GTiff/GeoTIFF
Size is 512, 512
Coordinate System is:
PROJCRS["WGS 84 / UTM zone 36N", ...]
Origin = (551180.000000000000000,3448840.000000000000000)
Pixel Size = (10.000000000000000,-10.000000000000000)
Corner Coordinates:
Upper Left  (  551180.000, 3448840.000) ( 33d32'13.46"E, 31d10'20.88"N)
Lower Left  (  551180.000, 3443720.000) ( 33d32'12.52"E, 31d 7'34.58"N)
Upper Right (  556300.000, 3448840.000) ( 33d35'26.87"E, 31d10'20.03"N)
Lower Right (  556300.000, 3443720.000) ( 33d35'25.84"E, 31d 7'33.73"N)
Center      (  553740.000, 3446280.000) ( 33d33'49.67"E, 31d 8'57.31"N)
Band 1 Block=512x8 Type=UInt16, ColorInterp=Gray
```

## Benchmark Summary

### Compression Performance

| Test | Input | FLAC Size | vs Raw | Lossless |
|------|-------|-----------|--------|----------|
| Full B04 tile | 10980x10980 uint16 | 116 MB | 50% | YES |
| 1024x1024 window | uint16 | 1.3 MB | 63% | YES |
| RGB (3 bands) | 512x512x3 uint16 | 1.2 MB | 60% | YES |

### Streaming Performance

| Operation | Download Size | Full File | Savings |
|-----------|---------------|-----------|---------|
| Center tile | 452 KB | 152 MB | 99.7% |
| Single tile (ID) | 460 KB | 152 MB | 99.7% |
| 484 tiles total | 152 MB | 152 MB | 0% |

### Key Findings

1. **Lossless Compression**: FLAC achieves 50% compression on raw Sentinel-2 uint16 data with zero data loss
2. **Metadata Preservation**: CRS, bounds, transform, nodata all preserved perfectly
3. **Streaming Efficiency**: Single tile extraction saves 99.7% bandwidth
4. **GDAL Compatible**: Reconstructed TIFFs are fully GDAL-compatible

## Multi-Band Example (RGB)

```bash
# Convert 3-band RGB to FLAC
flac-raster convert sentinel2_rgb.tif -o sentinel2_rgb.flac

# Verify round-trip
flac-raster convert sentinel2_rgb.flac -o sentinel2_rgb_back.tif
flac-raster compare sentinel2_rgb.tif sentinel2_rgb_back.tif
```

Output:
```
Arrays Equal: YES
Max Difference: 0.000000
```

## Use Cases

1. **Archival Storage**: 50% size reduction with guaranteed lossless quality
2. **Web Streaming**: HTTP range requests for tile-based access (99%+ bandwidth savings)
3. **Cloud Distribution**: Serve from any static file host (S3, CDN, etc.)
4. **Offline Analysis**: Small FLAC files for field work, expand to TIFF when needed

## Notes

- FLAC compression works best on spatially correlated data (satellite imagery, DEMs)
- Streaming format has overhead (~30% larger than standard FLAC) but enables partial downloads
- All geospatial metadata embedded in FLAC - no sidecar files needed
- Works with any GDAL-supported input format
