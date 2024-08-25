
from .two_flavor_matter import TwoFlavor
from .three_flavor_matter import ThreeFlavor

'''
TODO:

make animation for increasing baseline length, all 9 plots

write argparse examples. Write extensively
write examples that uses this package.
write constant density approximation code.
time this package vs other modules out there for python.

add job to build sdist as well - DONE: TODO: TEST THIS OUT
publish to conda as well
add github badges - build, version, etc.

document cython code as well.
use sphinx to build documentation.
host on readthedocs.io

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

document 'flavor indices' convention

add equation references in docstrings and comments in files


document two ways to install package from sdist:
python setup.py build
python setup.py install 

and

python -m build bla bla

also document how user needs working c compiler

'''