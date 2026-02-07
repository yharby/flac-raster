"""
Unified normalization module for FLAC-Raster.

Handles conversion between raster data types and audio sample formats
with proper precision handling for all common satellite imagery data types.

Supported data types:
- uint8: 8-bit unsigned (0-255) - RGB composites, classification
- int8: 8-bit signed (-128 to 127) - rare
- uint16: 16-bit unsigned (0-65535) - Sentinel-2, Landsat (most common)
- int16: 16-bit signed - DEMs, temperature, indices
- uint32: 32-bit unsigned - rare, high-precision counts
- int32: 32-bit signed - accumulated values
- float32: 32-bit float - reflectance, NDVI, processed data
- float64: 64-bit float - scientific analysis
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger("flac_raster.normalization")


@dataclass
class NormalizationParams:
    """Parameters needed for reversible normalization."""

    data_min: float
    data_max: float
    original_dtype: str
    bits_per_sample: int
    scale_factor: int  # The integer scale factor used (32767 for 16-bit, 8388607 for 24-bit)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "data_min": self.data_min,
            "data_max": self.data_max,
            "original_dtype": self.original_dtype,
            "bits_per_sample": self.bits_per_sample,
            "scale_factor": self.scale_factor,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizationParams":
        """Create from dictionary."""
        return cls(
            data_min=d["data_min"],
            data_max=d["data_max"],
            original_dtype=d["original_dtype"],
            bits_per_sample=d["bits_per_sample"],
            scale_factor=d.get("scale_factor", 32767),
        )


def get_dtype_info(dtype: np.dtype) -> Tuple[float, float, bool]:
    """
    Get the theoretical range and signedness for a numpy dtype.

    Returns:
        (min_value, max_value, is_integer)
    """
    dtype = np.dtype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max), True
    elif np.issubdtype(dtype, np.floating):
        # Float types don't have a fixed range
        return None, None, False
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def calculate_audio_params(data: np.ndarray, dtype: np.dtype) -> Tuple[int, int]:
    """
    Calculate appropriate sample rate and bit depth for FLAC encoding.

    Args:
        data: The raster data array
        dtype: The original data type

    Returns:
        (sample_rate, bits_per_sample)
    """
    dtype = np.dtype(dtype)

    # Determine bit depth based on data type precision requirements
    if dtype in (np.uint8, np.int8):
        # 8-bit data can be perfectly represented in 16-bit
        bits_per_sample = 16
    elif dtype in (np.uint16, np.int16):
        # 16-bit data needs 16-bit FLAC
        bits_per_sample = 16
    elif dtype in (np.uint32, np.int32, np.float32, np.float64):
        # Larger types need 24-bit for better precision
        # Note: pyFLAC accepts 32-bit input but encodes to 24-bit max
        bits_per_sample = 24
    else:
        logger.warning(f"Unknown dtype {dtype}, defaulting to 24-bit")
        bits_per_sample = 24

    # Calculate sample rate based on data size
    # Higher resolution data gets higher sample rate for FLAC's predictor
    if data.ndim >= 2:
        total_pixels = data.shape[-2] * data.shape[-1]
    else:
        total_pixels = data.size

    if total_pixels < 1_000_000:  # < 1MP
        sample_rate = 44100
    elif total_pixels < 10_000_000:  # < 10MP
        sample_rate = 48000
    elif total_pixels < 100_000_000:  # < 100MP
        sample_rate = 96000
    else:
        sample_rate = 192000

    logger.debug(f"Audio params for {dtype}: {sample_rate}Hz, {bits_per_sample}-bit")
    return sample_rate, bits_per_sample


def normalize_to_audio(
    data: np.ndarray,
    bits_per_sample: int,
    data_min: float = None,
    data_max: float = None,
) -> Tuple[np.ndarray, NormalizationParams]:
    """
    Normalize raster data to audio sample range.

    This function maps any raster data to the integer range expected by FLAC.
    The mapping is: data -> [-1, 1] -> [-scale_factor, scale_factor]

    Args:
        data: Input raster data (any dtype)
        bits_per_sample: Target bit depth (16 or 24)
        data_min: Override minimum value (optional, auto-detected if None)
        data_max: Override maximum value (optional, auto-detected if None)

    Returns:
        (audio_data, normalization_params)
    """
    original_dtype = str(data.dtype)

    # Compute data range if not provided
    if data_min is None:
        data_min = float(np.nanmin(data))
    if data_max is None:
        data_max = float(np.nanmax(data))

    # Handle edge case where all values are the same
    if data_max <= data_min:
        logger.warning(f"Data has no range (min={data_min}, max={data_max}), using zeros")
        data_range = 1.0  # Avoid division by zero
    else:
        data_range = data_max - data_min

    # Normalize to [-1, 1] range
    # Formula: normalized = 2 * (data - min) / (max - min) - 1
    data_float = data.astype(np.float64)
    data_norm = 2.0 * (data_float - data_min) / data_range - 1.0

    # Clip to ensure we're in valid range (handles NaN edge cases)
    data_norm = np.clip(data_norm, -1.0, 1.0)

    # Handle NaN values - replace with 0 (center of range)
    nan_mask = np.isnan(data_norm)
    if np.any(nan_mask):
        logger.warning(f"Found {np.sum(nan_mask)} NaN values, replacing with 0")
        data_norm[nan_mask] = 0.0

    # Scale to integer range
    if bits_per_sample == 16:
        scale_factor = 32767
        audio_data = (data_norm * scale_factor).astype(np.int16)
    elif bits_per_sample == 24:
        # pyFLAC uses 32-bit integers for 24-bit data
        scale_factor = 8388607
        audio_data = (data_norm * scale_factor).astype(np.int32)
    else:
        # Fallback to 32-bit range
        scale_factor = 2147483647
        audio_data = (data_norm * scale_factor).astype(np.int32)

    params = NormalizationParams(
        data_min=data_min,
        data_max=data_max,
        original_dtype=original_dtype,
        bits_per_sample=bits_per_sample,
        scale_factor=scale_factor,
    )

    logger.debug(
        f"Normalized {original_dtype} [{data_min:.4f}, {data_max:.4f}] -> "
        f"int{bits_per_sample} [{audio_data.min()}, {audio_data.max()}]"
    )

    return audio_data, params


def denormalize_from_audio(
    audio_data: np.ndarray,
    params: NormalizationParams,
) -> np.ndarray:
    """
    Convert audio data back to original raster format.

    This reverses the normalize_to_audio operation.

    Args:
        audio_data: Audio samples from FLAC decoder
        params: The normalization parameters used during encoding

    Returns:
        Reconstructed raster data in original dtype
    """
    # Determine the scale factor based on audio data type
    if audio_data.dtype == np.int16:
        scale_factor = 32767.0
    elif audio_data.dtype == np.int32:
        # Could be 24-bit (8388607) or 32-bit (2147483647)
        # Use the stored scale factor
        scale_factor = float(params.scale_factor)
    elif audio_data.dtype in (np.float32, np.float64):
        # Already normalized
        scale_factor = 1.0
    else:
        scale_factor = float(params.scale_factor)

    # Convert to normalized range [-1, 1]
    data_norm = audio_data.astype(np.float64) / scale_factor

    # Denormalize to original range
    # Formula: data = (normalized + 1) / 2 * (max - min) + min
    data_range = params.data_max - params.data_min
    data_float = (data_norm + 1.0) / 2.0 * data_range + params.data_min

    # Convert back to original dtype
    original_dtype = np.dtype(params.original_dtype)

    if np.issubdtype(original_dtype, np.integer):
        # Round to nearest integer
        data_out = np.round(data_float).astype(original_dtype)
    else:
        data_out = data_float.astype(original_dtype)

    logger.debug(f"Denormalized to {original_dtype}: [{data_out.min():.4f}, {data_out.max():.4f}]")

    return data_out


def estimate_precision_loss(
    original_dtype: np.dtype,
    data_min: float,
    data_max: float,
    bits_per_sample: int,
) -> dict:
    """
    Estimate the maximum precision loss for a given conversion.

    This helps users understand the trade-off between compression and accuracy.

    Returns:
        Dictionary with precision metrics
    """
    dtype = np.dtype(original_dtype)
    data_range = data_max - data_min

    if bits_per_sample == 16:
        quantization_levels = 65534  # 2 * 32767
    elif bits_per_sample == 24:
        quantization_levels = 16777214  # 2 * 8388607
    else:
        quantization_levels = 4294967294  # 2 * 2147483647

    # Maximum quantization error
    max_error = data_range / quantization_levels

    # Relative error as percentage
    if data_range > 0:
        relative_error_pct = (max_error / data_range) * 100
    else:
        relative_error_pct = 0.0

    # For integer types, check if we can be truly lossless
    is_lossless = False
    if np.issubdtype(dtype, np.integer):
        dtype_info = np.iinfo(dtype)
        dtype_range = dtype_info.max - dtype_info.min
        # Lossless if dtype range fits within our quantization levels
        is_lossless = dtype_range <= quantization_levels

    return {
        "max_absolute_error": max_error,
        "relative_error_percent": relative_error_pct,
        "quantization_levels": quantization_levels,
        "is_lossless": is_lossless,
        "bits_per_sample": bits_per_sample,
    }
