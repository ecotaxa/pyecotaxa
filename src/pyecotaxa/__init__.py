from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyecotaxa")
except PackageNotFoundError:
    # Fallback for running directly from an unpackaged source tree.
    __version__ = "0+unknown"
