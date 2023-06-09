
try:
    import numpy as np
    from .demo import pysolvers
    from .demo import utils
except ImportError:
    pass

from .two_flavor_matter import TwoFlavor
from .three_flavor_matter import ThreeFlavor

'''
TODO:

make animation for increasing baseline length, all 9 plots
configure argparse

exposed right now:

Eigen:
    probability
    probmatrix

CayleyHamilton:
    probability
    probmatrix

Identities:
    probability
    probmatrix

    deltamsq_matter
    angles_phase_matter
    PMNS_matter
'''

'''


document two ways to install package from sdist:
python setup.py build
python setup.py install 

and

python -m build bla bla

also document how user needs working c compiler

'''