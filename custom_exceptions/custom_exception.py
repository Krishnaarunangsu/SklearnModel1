# define python user-defined exceptions
class Error(Exception):
    """
    Base Class for other Exceptions
    """
    pass


class FileDetailsNotExist(Error):
    """
    File Details does not exist
    """
    pass


class ValueTooSmallError(Error):
    """
    Raised when the value is too small
    """
    pass


class ValueTooLargeError(Error):
    """
    Raised when the value is too small
    """
    pass
