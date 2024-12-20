import argparse
from . import version

def main():
    parser = argparse.ArgumentParser(description='⚗️drAIn CLI ⚗️')

    subparsers = parser.add_subparsers(dest='command', 
                                       help='sub-command help')
    parser.add_argument('-v', '--version', 
                        help="Print version of module.",
                        action='store_true')
    # data_parser = subparsers.add_parser('data')
    # data_parser.add_argument('-o', '--overview', help="Run the preconfigured overview of data.", action='store_true')

    args = parser.parse_args()

    if args.version:
        print(version.get_version(reduced=False))