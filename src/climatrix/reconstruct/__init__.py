import logging

log = logging.getLogger(__name__)

try:
    from .idw import IDWReconstructor as IDWReconstructor
except ImportError:
    log.warning(
        "IDWReconstructor is not available. Please install climatrix with idw support."
    )
    pass
try:
    from .kriging import (
        OrdinaryKrigingReconstructor as OrdinaryKrigingReconstructor,
    )
except ImportError:
    log.warning(
        "OrdinaryKrigingReconstructor is not available. Please install climatrix with kriging support."
    )
    pass
try:
    from .sinet import SiNETReconstructor as SiNETReconstructor
except ImportError:
    log.warning(
        "SiNETReconstructor is not available. Please install climatrix with sinet support."
    )
    pass
try:
    from .mmgn import MMGNReconstructor as MMGNReconstructor
except ImportError:
    log.warning(
        "MMGNReconstructor is not available. Please install climatrix with mmgn support."
    )
    pass
