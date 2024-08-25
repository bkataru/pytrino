import argparse
from math import radians

from .two_flavor_matter import TwoFlavor
from .three_flavor_matter import ThreeFlavor

def main():
    description = "Compute neutrino oscillation probabiltiies at the speed of C" # TODO refine this

    parser = argparse.ArgumentParser(prog="pytrino", description=description, formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='flavors', help='No. of flavors to consider: twoflavor, threeflavor', metavar='flavors')

    parser_twoflavor = subparsers.add_parser('twoflavor', description="Compute two flavor oscillation probabilities in vacuum/matter", formatter_class=argparse.RawTextHelpFormatter) # TODO document how we only need pee and pemu for two flavor, can compute 21 and 22 from these.
    parser_threeflavor = subparsers.add_parser('threeflavor', description="Compute three flavor oscillation probabilities vacuum/matter", formatter_class=argparse.RawTextHelpFormatter)

    parser_twoflavor.add_argument('delmsq', type=float, help='Decimal value for delmsq (in eV^2): the two-flavor mass squared difference in vacuum')
    parser_twoflavor.add_argument('theta', type=float, help='Decimal value for theta (in degrees): the two-flavor mixing angle in vacuum')

    probhelp = '''Probability to compute (1 = e, 2 = mu, 3 = tau): 
            11: e -> e 
            12: e -> mu 
            13: e -> tau 
            21: mu -> e 
            22: mu -> mu 
            23: mu -> tau 
            31: tau -> e 
            32: tau -> mu 
            33: tau -> tau 
            all: Compute probability matrix containing all survival and transition probabilities'''

    parser_threeflavor.add_argument('probability', metavar='probability', choices=['11', '12', '13', '21', '22', '23', '31', '32', '33', 'all'],
                                        help=probhelp)
    parser_threeflavor.add_argument('delmsq21', type=float, help='Decimal value for delmsq21 (in eV^2): the 21 mass squared difference in vacuum')
    parser_threeflavor.add_argument('delmsq31', type=float, help='Decimal value for delmsq31 (in eV^2): the 31 mass squared difference in vacuum')
    parser_threeflavor.add_argument('deltacp', type=float, help='Decimal value for deltacp (in degrees): the Dirac-type CP violating phase in vacuum')
    parser_threeflavor.add_argument('theta12', type=float, help='Decimal value for theta12 (in degrees): the mixing angle in 12 in vacuum')
    parser_threeflavor.add_argument('theta13', type=float, help='Decimal value for theta13 (in degrees): the mixing angle in 13 in vacuum')
    parser_threeflavor.add_argument('theta23', type=float, help='Decimal value for theta23 (in degrees): the mixing angle in 23 in vacuum')

    parser_threeflavor.add_argument(
        '-l', 
        '--labels', 
        action='store_true', 
        help='Include reference labels for probabilities when outputting the probability matrix'
    )

    for psr in [parser_twoflavor, parser_threeflavor]:
        psr.add_argument(
            '-b', 
            '--baseline',
            metavar='', 
            type=float, 
            default=1300,
            help='''Decimal value for baseline (in km): neutrino beam travel distance \n(default: %(default)d km)'''
        )

        psr.add_argument(
            '-e', 
            '--energy',
            metavar='', 
            type=float, 
            default=0.001, 
            help='''Decimal value for energy (in GeV): neutrino beam energy \n(default: %(default)f GeV)'''
        )

        psr.add_argument(
            '-v', 
            '--V',
            metavar='', 
            type=float, 
            default=0, 
            help='''Decimal value for V (in eV): effective constant matter potential \n(default: %(default)d eV)'''
        )

        psr.add_argument(
            '-a', 
            '--antineutrinos',
            action='store_true', 
            help='Compute probabilities for antineutrino transitions instead'
        )

    args = parser.parse_args()

    flavors = args.flavors
    if flavors == 'twoflavor':
        constants = [args.delmsq, radians(args.theta)]
        solver = TwoFlavor(*constants)

        Pee, Pemu = solver.probability(args.baseline, args.energy, args.V, args.antineutrinos)

        print(f"Pee: {Pee}")
        print(f"Pemu: {Pemu}")

    elif flavors == 'threeflavor': 
        # Create a list of constants
        constants = [args.delmsq21, args.delmsq31, args.deltacp, radians(args.theta12), radians(args.theta13), radians(args.theta23)]

        # Create an instance of the ThreeFlavor class with the constants
        solver = ThreeFlavor(*constants)

        prob = args.probability

        if prob == 'all':
            result = solver.probmatrix(args.baseline, args.energy, args.V, args.antineutrinos, args.labels)

            print(result)
        else:
            i, j = [int(d) for d in prob]
            labelmatrix = [["Pee", "Pemu", "Petau"], ["Pmue", "Pmumu", "Pmutau"], ["Ptaue", "Ptaumu", "Ptautau"]]
            label = labelmatrix[i - 1][j - 1]

            result = solver.probability(i, j, args.baseline, args.energy, args.V, args.antineutrinos)

            print(f"{label}: {result}") 


if __name__ == '__main__':
    main()