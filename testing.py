import numpy as np

from pytrino import oscprobs

# a = 0.2
# theta12 = np.radians(33.2)
# theta13 = np.radians(4.4)
# theta23 = np.radians(46.1)
# alpha = 0.026
# delta = 50
# deltacp = np.pi/6

# baseline = 10
# energy = 2 * 1e-3
# delmsq31 = 2.5e-3
# delmsq21 = 7.55e-5
# deltacp = 0 # np.pi/6
# theta13 = np.pi/20
# theta12 = np.pi/6
# theta23 = np.pi/4
# V = 0

# prob1 = Eigen(baseline, energy, V, delmsq21, delmsq31, deltacp, theta12, theta13, theta23)
# prob2 = CayleyHamilton(baseline, energy, V, delmsq21, delmsq31, deltacp, theta12, theta13, theta23)
# prob3 = Identities(baseline, energy, 0, delmsq21, delmsq31, deltacp, theta12, theta13, theta23)

# # print(prob1.probabilities().tolist())
# # print(prob2.probabilities().tolist())
# # print(prob3.probabilities().tolist())

# print(prob3.appearance(1, 2))

print(oscprobs.Identities(10, 0.002, 0, 7.55e-05, 0.0025, 0, 0.5235987755982988, 0.15707963267948966, 0.7853981633974483).appearance(1, 2))

# import sys
# sys.exit()

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
