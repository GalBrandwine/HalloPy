"""This utils.py module holds the location of the project root path,

for further file opening in the project.
"""
from os import path as op

MAIN_DIRECTORY = op.dirname(op.dirname(__file__))
print("MAIN_DIRECTORY: {}".format(MAIN_DIRECTORY))


def get_full_path(*path):
    """Function returns file path, relative to project root.  """
    return op.join(MAIN_DIRECTORY, *path)
