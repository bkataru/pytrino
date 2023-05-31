![](logo.png)

We use the recently discovered Eigenvalue-Eigenvector and Adjugate Identities to compute oscillation probabilities with minimal algorithmic steps.

# Usage

## Imports
```Python
import numpy as np
np.set_printoptions(precision=3)

from pytrino.oscprobs import OscProbIdentities
```

## Define your oscillation parameters

```Python
a = 0.2
theta12 = np.radians(33.2)
theta13 = np.radians(4.4)
theta23 = np.radians(46.1)
alpha = 0.026
delta = 50
deltacp = np.pi/6
```

## Compute probabilities
```Python
prob = OscProbIdentities(alpha, a, delta, deltacp, theta12, theta13, theta23)

print(prob.mat_angles_phase()) # prints mixing angles and phase in matter
# (29.988882340423384, 84.48628984963665, 3.5930259417910695, 44.188561816129464)

print(prob.PMNS()) # prints values of PMNS matrix elements
"""
[[ 0.849+0.000e+00j -0.297+0.000e+00j -0.062-4.318e-01j]
 [-0.335+8.346e-02j -0.92 -2.921e-02j -0.184+1.342e-18j]
 [ 0.01 +3.989e-01j  0.214-1.396e-01j -0.881-7.870e-19j]]
"""

print(prob.PMNS() @ prob.PMNS().H) # unitarity check
"""
[[1.000e+00-1.696e-22j 0.000e+00+1.561e-17j 6.939e-18+5.551e-17j]
 [0.000e+00-1.561e-17j 1.000e+00-4.276e-19j 5.551e-17-1.327e-18j]
 [1.388e-17-4.857e-17j 5.551e-17+0.000e+00j 1.000e+00-8.097e-19j]]
"""

print(prob.probabilities()) # prints all 9 probabilities
"""
[[0.97  0.015 0.014]
 [0.014 0.236 0.75 ]
 [0.015 0.749 0.236]]
"""
```