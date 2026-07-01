import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import transitleastsquares
print(transitleastsquares.transitleastsquares.power.__code__.co_varnames)
