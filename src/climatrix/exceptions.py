class DatasetCreationError(ValueError):
    pass


class TimeRelatedDatasetNotSupportedError(DatasetCreationError):
    pass


class TooLargeSamplePortionWarning(UserWarning):
    pass


class LongitudeConventionMismatch(ValueError):
    pass
