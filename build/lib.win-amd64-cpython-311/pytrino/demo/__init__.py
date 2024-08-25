try:
    import numpy as np
    
    NUMPY_ENABLED = True
except ImportError:
    NUMPY_ENABLED = False

if not NUMPY_ENABLED:

    err = '''
    Error: NumPy is not installed. Please install NumPy before using the demo package. 
    
    To install NumPy, open a terminal and run the following command:

    pip install numpy

    Once NumPy is installed, you can try importing from the demo package again.
    '''

    raise ImportError(err)
else:
    from . import pysolvers
    from . import utils