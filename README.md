
<p align="center" width="100%">
    <img width="75%" src="https://raw.githubusercontent.com/BK-Modding/pytrino/main/logo.png"> 
</p>

Use Python to compute neutrino oscillation probabilities in vacuum and matter at the speed of C!

We use the recently discovered Eigenvalue-Eigenvector and Adjugate Identities to compute oscillation probabilities with minimal algorithmic steps. Implemented using Cython to have the convenience of Python but the speed of C.

# Installation

Pytrino is published on PyPI and condaforge.

## Using pip

```console
pip install pytrino
```
or
```console
pip3 install pytrino
```
or
```console
python -m pip install pytrino
```

## Using conda

```console
conda install pytrino
```

## Building from source


### Method one

```console
python setup.py build
python setup.py install
```

### Method two

```console
python -m build
```

```console
pip install dist/pytrino-[version-no].tar.gz
```



Build wheels


# Usage

## Imports
```Python
from pytrino import oscprobs
```

## Define your oscillation parameters

```Python
baseline = 10 # baseline length L
energy = 2 * 1e-3 # neutrino beam energy E
delmsq31 = 2.5e-3 # mass squared difference 31 in vacuum
delmsq21 = 7.55e-5 # mass squared difference 21 in vacuum
deltacp = np.pi/6 # Dirac CP-violating phase in vacuum
theta13 = np.pi/20 # mixing angle in vacuum
theta12 = np.pi/6 # mixing angle in vacuum
theta23 = np.pi/4 # mixing angle in vacuum
V = 0 # matter (MSW) effect potential
```

## Instantiate probability solver object
Probabilities via diagonalizing Hamiltonian in matter

```Python
probsolver = Eigen(baseline, energy, V, delmsq21, delmsq31, deltacp, theta12, theta13, theta23)
```

```bash
pip install pytrino

Probabilities via Cayley-Hamilton formalism
```Python
probsolver = CayleyHamilton(baseline, energy, V, delmsq21, delmsq31, deltacp, theta12, theta13, theta23)
```

Probabilities using Linear Algebra identities
```Python
probsolver = Identities(baseline, energy, 0, delmsq21, delmsq31, deltacp, theta12, theta13, theta23)
```

## Compute probabilities
```Python

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


# Contributions

Please open a pull request if you have any improvements/changes that might benefit the package.

Please open an issue if there is any difficulty installing/using the package.

# Contact me