class SertitException(Exception):
    """A base class for Sertit exceptions."""


class ListCondaEnvError(SertitException):
    """Raise this exception if one failed to list conda environment"""
