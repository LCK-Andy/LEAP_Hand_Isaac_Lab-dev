"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions (only works inside Omniverse Kit).
try:
    from .ui_extension_example import *
except ImportError:
    pass