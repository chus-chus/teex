""" Utils for dataset handling """

import sys


def query_yes_no(question, default="yes", _expectInput=True): # pragma: no cover
    """Ask a yes/no question via raw_input() and return their answer.

    :param str question: is a string that is presented to the user.
    :param str default: is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
    :param bool _expectInput: Only for testing usage.

    :return: True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    
    if not _expectInput:
        if default is not None:
            return valid[default]
        else:
            raise ValueError("default must not be None when not \
                             expecting input")

    while True:
        sys.stdout.write(question + prompt)
        try:
            choice = input().lower()
        except EOFError:
            choice = ""
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")