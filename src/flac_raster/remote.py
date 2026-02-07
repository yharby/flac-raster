"""
Remote file access module for FLAC-Raster.

Provides unified access to remote files via:
- HTTP/HTTPS URLs (via requests)
- S3 URLs (s3://bucket/key) via obstore
- Azure Blob (az://container/blob) via obstore
- Google Cloud Storage (gs://bucket/key) via obstore

Supports HTTP range requests for efficient partial downloads.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger("flac_raster.remote")

# Try to import obstore for cloud storage support
try:
    from obstore.store import AzureStore, GCSStore, S3Store

    OBSTORE_AVAILABLE = True
except ImportError:
    OBSTORE_AVAILABLE = False
    logger.debug("obstore not available - cloud storage features disabled")


def is_remote_url(path: Union[str, Path]) -> bool:
    """Check if path is a remote URL."""
    if isinstance(path, Path):
        return False
    path_str = str(path)
    return path_str.startswith(("http://", "https://", "s3://", "az://", "gs://"))


def get_url_scheme(url: str) -> str:
    """Get the scheme from a URL."""
    parsed = urlparse(url)
    return parsed.scheme.lower()


def parse_cloud_url(url: str) -> Tuple[str, str, str]:
    """
    Parse a cloud storage URL into (scheme, bucket, key).

    Supports:
    - s3://bucket/path/to/file.tif
    - az://container/path/to/file.tif
    - gs://bucket/path/to/file.tif
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return scheme, bucket, key


class RemoteFile:
    """
    Remote file access with HTTP range request support.

    Works with HTTP/HTTPS URLs and cloud storage (S3, Azure, GCS).
    """

    def __init__(self, url: str):
        """
        Initialize remote file access.

        Args:
            url: Remote URL (http://, https://, s3://, az://, gs://)
        """
        self.url = url
        self.scheme = get_url_scheme(url)
        self._store = None
        self._file_size: Optional[int] = None

        if self.scheme in ("http", "https"):
            self._init_http()
        elif self.scheme in ("s3", "az", "gs"):
            self._init_cloud()
        else:
            raise ValueError(f"Unsupported URL scheme: {self.scheme}")

    def _init_http(self):
        """Initialize HTTP/HTTPS access."""
        import requests

        # Get file size via HEAD request
        try:
            response = requests.head(self.url, timeout=10)
            response.raise_for_status()
            self._file_size = int(response.headers.get("content-length", 0))
            self._supports_range = "bytes" in response.headers.get("accept-ranges", "").lower()
        except Exception as e:
            logger.warning(f"Failed to get file info: {e}")
            self._supports_range = True  # Assume it works

    def _init_cloud(self):
        """Initialize cloud storage access."""
        if not OBSTORE_AVAILABLE:
            raise ImportError(
                "obstore is required for cloud storage access. Install with: pip install obstore"
            )

        scheme, bucket, self._key = parse_cloud_url(self.url)

        if scheme == "s3":
            # S3 - will use environment credentials or instance profile
            self._store = S3Store(bucket=bucket)
        elif scheme == "az":
            self._store = AzureStore(container=bucket)
        elif scheme == "gs":
            self._store = GCSStore(bucket=bucket)

    @property
    def file_size(self) -> Optional[int]:
        """Get the total file size in bytes."""
        if self._file_size is not None:
            return self._file_size

        if self.scheme in ("http", "https"):
            return self._file_size  # Already fetched in init

        if self._store is not None:
            try:
                meta = self._store.head(self._key)
                self._file_size = meta.size
                return self._file_size
            except Exception as e:
                logger.warning(f"Failed to get cloud file size: {e}")

        return None

    def read_range(self, start: int, end: int) -> bytes:
        """
        Read a byte range from the remote file.

        Args:
            start: Start byte (inclusive)
            end: End byte (inclusive)

        Returns:
            Bytes from the specified range
        """
        if self.scheme in ("http", "https"):
            return self._read_http_range(start, end)
        else:
            return self._read_cloud_range(start, end)

    def _read_http_range(self, start: int, end: int) -> bytes:
        """Read a byte range via HTTP range request."""
        import requests

        headers = {"Range": f"bytes={start}-{end}"}
        response = requests.get(self.url, headers=headers, timeout=60)

        if response.status_code == 206:  # Partial content
            return response.content
        elif response.status_code == 200:
            # Server doesn't support range requests, got full content
            logger.warning("Server returned full content, extracting range")
            return response.content[start : end + 1]
        else:
            response.raise_for_status()
            return response.content

    def _read_cloud_range(self, start: int, end: int) -> bytes:
        """Read a byte range from cloud storage."""
        if self._store is None:
            raise RuntimeError("Cloud store not initialized")

        # obstore uses (start, end) as exclusive range
        data = self._store.get_range(self._key, start=start, end=end + 1)
        return bytes(data)

    def read_all(self) -> bytes:
        """Read the entire file."""
        if self.scheme in ("http", "https"):
            import requests

            response = requests.get(self.url, timeout=120)
            response.raise_for_status()
            return response.content
        else:
            if self._store is None:
                raise RuntimeError("Cloud store not initialized")
            data = self._store.get(self._key)
            return bytes(data)

    def download_to_temp(self) -> Path:
        """
        Download the file to a temporary location.

        Returns:
            Path to the temporary file
        """
        suffix = Path(urlparse(self.url).path).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(self.read_all())
            return Path(tmp.name)


def open_remote(url: str) -> RemoteFile:
    """
    Open a remote file for reading.

    Args:
        url: Remote URL (http://, https://, s3://, az://, gs://)

    Returns:
        RemoteFile instance
    """
    return RemoteFile(url)


def read_remote_range(url: str, start: int, end: int) -> bytes:
    """
    Convenience function to read a byte range from a remote URL.

    Args:
        url: Remote URL
        start: Start byte (inclusive)
        end: End byte (inclusive)

    Returns:
        Bytes from the specified range
    """
    remote = RemoteFile(url)
    return remote.read_range(start, end)


def download_remote(url: str, output_path: Optional[Path] = None) -> Path:
    """
    Download a remote file.

    Args:
        url: Remote URL
        output_path: Optional output path (uses temp file if not specified)

    Returns:
        Path to the downloaded file
    """
    remote = RemoteFile(url)

    if output_path is None:
        return remote.download_to_temp()

    data = remote.read_all()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(data)

    return output_path
