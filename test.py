
import numpy as np
from pytrino.oscprobs import Identities
 
a = 0.2 # fix this, change to V and delmsq21, delmsq31
theta12 = np.radians(33.2)
theta13 = np.radians(4.4)
theta23 = np.radians(46.1)
alpha = 0.026
delta = 50
deltacp = np.pi/6

prob = Identities(alpha, a, delta, deltacp, theta12, theta13, theta23)

np.set_printoptions(precision=3)

print(prob.mat_angles_phase())

print(prob.PMNS())

print(prob.PMNS() @ prob.PMNS().H) # unitarity check

print(prob.probabilities())

# print(prob1.mat_angles_phase())



# a = 0.2
# theta13 = np.pi/20
# theta12 = np.pi/6
# theta23 = np.pi/4
# deltacp = np.pi/6
# alpha = 0.03
# delta = lambda En: (1.27 * (2e-3) * 1e+3)/En

# # prob1 = OscProbIdentities(alpha, a, delta(3), deltacp, theta12, theta13, theta23)
# # prob2 = OscProbStandard(alpha, a, delta(3), deltacp, theta12, theta13, theta23)
# # prob3 = OscProbCayleyHamilton(alpha, a, delta(3), deltacp, theta12, theta13, theta23)

# # print(prob1.prob_appearance(1, 2))
# # print(prob2.prob_appearance(1, 2))
# # print(prob3.prob_appearance(1, 2))

# energies = np.linspace(0.1, 5, 1000)

# probs = [[], [], []]
# for en in energies:
#     prob1 = OscProbStandard(alpha, a, delta(en), deltacp, theta12, theta13, theta23)
#     prob2 = OscProbCayleyHamilton(alpha, a, delta(en), deltacp, theta12, theta13, theta23)
#     prob3 = OscProbIdentities(alpha, a, delta(en), deltacp, theta12, theta13, theta23)

#     probs[0].append(prob1.prob_appearance(2, 3))
#     probs[1].append(prob2.prob_appearance(2, 3))
#     probs[2].append(prob3.prob_appearance(2, 3))

# import matplotlib.pyplot as plt

# plt.plot(energies, probs[0])
# plt.plot(energies, probs[1])
# plt.plot(energies, probs[2])
# plt.show()
