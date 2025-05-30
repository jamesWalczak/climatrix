class DatasetCreationError(ValueError):
    pass


class TimeRelatedDatasetNotSupportedError(DatasetCreationError):
    pass


class LongitudeConventionMismatch(ValueError):
    pass


class MissingAxisError(KeyError):
    pass


class SubsettingByNonDimensionAxisError(ValueError):
    pass


class AxisMatchingError(ValueError):
    pass


class DomainMismatchError(ValueError):
    pass
