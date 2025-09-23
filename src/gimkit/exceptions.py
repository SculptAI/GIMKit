class GIMError(Exception):
    """Base exception class for GIM-related errors."""


class InvalidFormatError(GIMError):
    """Exception raised for invalid GIM query/response format."""
