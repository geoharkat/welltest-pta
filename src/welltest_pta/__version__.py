from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("welltest-pta")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.1.0"
