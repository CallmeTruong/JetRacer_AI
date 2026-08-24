# Urban Navigation Package Initialization
try:
    from jetracer.urban.config import DETECTION_CLASSES, ROUTE_COMMANDS, NUM_WAYPOINTS
except ImportError:
    from .config import DETECTION_CLASSES, ROUTE_COMMANDS, NUM_WAYPOINTS
