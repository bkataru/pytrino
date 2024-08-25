import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pytrino.demo.solvers import Identities, CayleyHamilton, Eigen

import pytrino

# np.set_printoptions(suppress=True)

# V = 0

# def round_complex_to_zero(arr, threshold):
#     real_part = np.where(np.abs(arr.real) < threshold, 0, arr.real)
#     imag_part = np.where(np.abs(arr.imag) < threshold, 0, arr.imag)
#     return real_part + 1j * imag_part

# numbers = np.array([1.23456e-12 + 5.4321e-13j, 3.45678e-14 + 7.6543e-15j, 1.23456e-16 + 1.23456e-16j, 1.0e-10 + 2.0e-11j])
# rounded_numbers = round_complex_to_zero(numbers, 1e-10)

# print(rounded_numbers)

# import sys
# sys.exit()

# import two_flavor_matter

# solver = two_flavor_matter.TwoFlavor(2.5e-3, 0.2 * pi)

# pee, pemu = solver.probability(1, 0.0000001)

# print(pee, pemu)

def cython_example():
    from pytrino import ThreeFlavor
    import numpy as np

    delmsq21 = 7.55e-5 
    delmsq31 = 2.5e-3
    deltacp = 1.32 * np.pi
    theta12 = np.radians(33.2)
    theta13 = np.radians(4.4)
    theta23 = np.radians(46.1)

    constants = [delmsq21, delmsq31, deltacp, theta12, theta13, theta23]

    baseline = 1000
    energy = 0.001
    V = 0 # 1e-14

    solver = ThreeFlavor(*constants)

    print(solver.probmatrix(baseline, energy, V))



def example1():
    delmsq31 = 2.5e-3
    delmsq21 = 7.55e-5 
    deltacp = 1.32 * np.pi
    theta13 = np.radians(4.4)
    theta12 = np.radians(33.2)
    theta23 = np.radians(46.1)

    constants = [delmsq21, delmsq31, deltacp, theta12, theta13, theta23]

    baseline = 1000
    energy = 0.001
    V = 0 # 1e-14
    
    # Pemu = Identities(*constants).probability(1, 2, baseline, energy, V)
    # Pmutau = Identities(*constants).probability(2, 3, baseline, energy, V)

    # _Pemu = Identities(*_constants).probability(1, 2, baseline, energy, V)
    # _Pmutau = Identities(*_constants).probability(2, 3, baseline, energy, V)

    # Pee = 1 - (Pemu + _Pemu)
    # Petau = _Pemu
    # Pmue = Pemu - Pmutau + _Pmutau
    # Pmumu = 1 - Pemu - _Pmutau
    # Ptaue = _Pemu + Pmutau - _Pmutau
    # Ptaumu = _Pmutau
    # Ptautau = 1 - (_Pemu + Pmutau)
    
    # pmatri = np.matrix([[Pee, Pemu, Petau], [Pmue, Pmumu, Pmutau], [Ptaue, Ptaumu, Ptautau]])

    # print(pmatri)

    # m1 = Identities(*constants).probmatrix(baseline, energy, V, antineutrinos=True)
    # solver = Identities(*constants)
    # m2 = solver.probmatrix(baseline, energy, V)
    # print(m2)
    # print(CayleyHamilton(*constants).probmatrix(baseline, energy, V))

    # # print(solver.mat_deltamsq(baseline, energy, V))
    # print("elem", np.abs(solver.PMNSmatter(baseline, energy, V)[1, 1])**2)
    # print()
    # print(solver.probability(2, 2, baseline, energy, V))
    # print(m2)

    # print(solver.PMNSmatter(baseline, energy, V))

    sv = Identities(*constants)

    print(sv.probmatrix(baseline, energy, V, labels=True, antineutrinos=True))
    # print("=" * 50)
    # print(Eigen(*constants).probmatrix(baseline, energy, V))
    # print(CayleyHamilton(*constants).probmatrix(baseline, energy, V))

    # print(Identities(*constants).probability(1, 1, baseline, energy, V))

    # print(m1 - m2)

    # from timeit import default_timer as timer

    # start = timer()
    # m1 = Identities(*constants).probmatrix(baseline, energy, V)
    # end = timer()
    # timediff = end - start
    # print("Using relations:", timediff * 1e+3) # Time in seconds, e.g. 5.38091952400282

    # print()

    # start = timer()
    # m2 = Identities(*constants).probmatrix(baseline, energy, V, use_relations=False)
    # end = timer()
    # timediff = end - start
    # print("Without relations:", timediff * 1e+3) # Time in seconds, e.g. 5.38091952400282

    # for i in range(9):
    #     for mat in [m1, m2]:
    #         mat = np.array(mat)
    #         print(mat.flatten()[i], end = " ")
    #     print()

example1()

def example2():
    delmsq31 = 2e-3
    delmsq21 = 7.55e-5
    deltacp = 1.32 * np.pi
    theta13 = np.radians(4.4)
    theta12 = np.radians(33.2)
    theta23 = np.radians(46.1)

    constants = [delmsq21, delmsq31, deltacp, theta12, theta13, theta23]

    baseline = 1090
    # energy = 0.001
    Vfn = lambda rho: 7.56e-14 * rho * 0.5
    V = Vfn(9)

    from timeit import default_timer as timer

    energies = np.linspace(0.1, 5, 1000)

    times = [] # Eigen, Cayley, Identities
    for energy in energies:
        timearr = []

        start = timer()
        m1 = Eigen(*constants).probmatrix(baseline, energy, V)
        end = timer()
        timediff = end - start

        timearr.append(timediff * 1e+3)
        # print("Eigen", timediff * 1e+3) 

        start = timer()
        m1 = CayleyHamilton(*constants).probmatrix(baseline, energy, V)
        end = timer()
        timediff = end - start

        timearr.append(timediff * 1e+3)
        # print("Cayley", timediff * 1e+3) 

        start = timer()
        m1 = Identities(*constants).probmatrix(baseline, energy, V)
        end = timer()
        timediff = end - start

        timearr.append(timediff * 1e+3)
        # print("Identities", timediff * 1e+3)

        times.append(timearr)

    times = np.array(times)

    plt.plot(energies, times[:, 0], label="Eigen")
    plt.plot(energies, times[:, 1], label="Cayley")
    plt.plot(energies, times[:, 2], label="Identities")

    plt.legend()
    plt.show()

# example2()

import sys
sys.exit()

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15, 15))

baseline_values = np.linspace(1, 1000, 1000)  # Baseline values for animation
energies = np.linspace(0.1, 5, 100)

V = 0

bprobs = []
for ind, bline in enumerate(baseline_values):
    probs = []
    for energy in energies:
        probmatrix = probsolver.probmatrix(bline, energy, V)

        probs.append(probmatrix)

    probs = np.array(probs)
    bprobs.append(probs)
    print(f"finished {ind}")

print("computed")

def plotgrid(frame):
    probs = bprobs[frame]
    for i in range(3):
        for j in range(3):
            prob = probs[:, i, j]
            
            axs[i, j].plot(energies, prob)

# # plotgrid(0)
# V = lambda rho: 7.56e-14 * rho * 0.5
# plotgrid(V(0))

def update(frame):
    labelmatrix = np.matrix([["Pee", "Pemu", "Petau"], ["Pmue", "Pmumu", "Pmutau"], ["Ptaue", "Ptaumu", "Ptautau"]])

    for i in range(3):
        for j in range(3):
            label = labelmatrix[i, j]
            axs[i, j].set_title(label)
    
    fig.suptitle(f'Probabilities (Baseline: {baseline_values[frame]} km)')

    for ax in axs.flat:
        ax.clear()
    plotgrid(frame)

animation = FuncAnimation(fig, update, frames=len(baseline_values), interval=1)
plt.show()

# probsolver = Eigen(baseline, energy, V, delmsq21, delmsq31, deltacp, theta12, theta13, theta23)
# probsolver = CayleyHamilton(baseline, energy, V, delmsq21, delmsq31, deltacp, theta12, theta13, theta23)

# # print(prob1.probabilities().tolist())
# # print(prob2.probabilities().tolist())
# # print(prob3.probabilities().tolist())

# print(prob3.appearance(1, 2))

# print(oscprobs.Identities(10, 0.002, 0, 7.55e-05, 0.0025, 0, 0.5235987755982988, 0.15707963267948966, 0.7853981633974483).appearance(1, 2))

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
