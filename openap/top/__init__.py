from .base import Base
from .climb import Climb
from .cruise import Cruise
from .descent import Descent
from .full import CompleteFlight, MultiPhase
from . import vis

try:
    from . import wind
except:
    RuntimeWarning("cfgrib and sklearn are required for wind integration")

# import warnings

# warnings.filterwarnings("ignore")
