"Used to create connection using credentials config file"

from .redshift import redshift
from .creds import create_connection, close_connection


def test_function():
    return "this worked!"
