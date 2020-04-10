import sys


def not_supported_param(param, value):
    raise Exception("You set the param {}={}, but not supported yet.".format(param, value))
    sys.exit(0)

