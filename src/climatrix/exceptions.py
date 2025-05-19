class DatasetCreationError(ValueError):
    pass


class TimeRelatedDatasetNotSupportedError(DatasetCreationError):
    pass


class LongitudeConventionMismatch(ValueError):
    pass


class MissingAxisError(KeyError):
    pass
