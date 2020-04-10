import sys


def not_supported_param(param, value):
    print("\nYou set the param {}={}, but not supported yet.".format(param, value))
    sys.exit(0)

