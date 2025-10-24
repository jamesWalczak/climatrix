class DatasetCreationError(ValueError):
    pass


class OperationNotSupportedForDynamicDatasetError(NotImplementedError):
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


class MissingDependencyError(ImportError):
    """Raised when required plotting dependencies are not installed."""

    pass


class ReconstructorConfigurationFailed(ValueError):
    """Raised when a reconstructor cannot be configured properly."""

    pass
