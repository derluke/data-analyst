class NumericCleaningError(Exception):
    """Raised when numeric cleaning fails"""

    pass


class DateCleaningError(Exception):
    """Raised when date parsing fails"""

    pass


class CategoryCleaningError(Exception):
    """Raised when categorical cleaning fails"""

    pass


class EmptyDataError(Exception):
    """Raised when cleaning results in empty data"""

    pass
