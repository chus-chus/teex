import os


def check_dir(pathName):
    if not os.path.isdir(pathName):
        raise Exception('Data path not valid: {}'.format(pathName))
    if len(os.listdir(pathName)) == 0:
        raise Exception('Empty directory: {}'.format(pathName))


def check_and_create_dir(pathName):
    if not os.path.exists(pathName):
        os.mkdir(pathName)


def check_file(pathName):
    if not os.path.isfile(pathName):
        raise Exception('File not valid: {}'.format(pathName))
