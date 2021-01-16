from sshtunnel import SSHTunnelForwarder
import pandas
import sshtunnel
import getpass
import configparser
import psycopg2
from pandas.io.sql import read_sql_query as sql
import keyring
import random

tunnel = None
conn = None

def config_from_keyring(config, service):
    for key in [k for k in config.options(service) if k != 'keyring']:
        config.set(service, key, keyring.get_password(service, key))

#TODO create function for rewritting config file
#def keyring_to_config(config, service)


def config_to_keyring(config, service, path):
    overwrite_config = configparser.ConfigParser()
    overwrite_config.read(path)
    for key in config.options(service):
        value = config.get(service, key)
        keyring.set_password(service, key, value)
        overwrite_config.set(service, key, '********')
    with open(path, 'w') as config_file:
        overwrite_config.set(service, 'keyring', 'True')
        overwrite_config.write(config_file)

def read_config(path='/Users/{username}/.poshmark.cfg'):
    username = getpass.getuser()
    path = path.format(username=username)
    config = configparser.ConfigParser()
    config.read(path)

    # check config for keyring flag. Read from keyring if present, store in keyring if missing.
    for service in config.sections():
        if 'keyring' in config.options(service):
            config_from_keyring(config, service)
        else:
            config_to_keyring(config, service, path)
    return config

def fetch_password(service='system', key=getpass.getuser(), update=False):
    password = keyring.get_password(service, key)
    if not password or update:
        password = getpass.getpass(prompt=f'Password for {service} {key}: ')
        keyring.set_password(service, key, password)
    return password

def _create_tunnel(config, port=random.choice(range(3000,7000))):
    # Create an SSH tunnel

    tunnel = SSHTunnelForwarder(
        (config['redshift']['tunnel_host'], int(config['redshift']['tunnel_port'])),
        ssh_username=config['redshift']['user'],
        #ssh_password = getpass.getpass(prompt='Password for ssh key: '),
        #ssh_private_key='/Users/{}/.ssh/id_rsa'.format(getpass.getuser()),
        remote_bind_address=(config['redshift']['host'], int(config['redshift']['port'])),
        local_bind_address=('localhost',port), # could be any available port
    )
    tunnel.SSH_TIMEOUT = 3600
    # Start the tunnel
    if (tunnel.is_active != True):
        tunnel.start()
    return tunnel

def tunnel_start():
    global tunnel
    tunnel.start()

def create_connection(conn_name='redshift', port = random.choice(range(3000,7000))):
    """
    Create connection to Postgres Database.
    # Arguments
        conn_name (str): Connection name found in .poshmark.cfg file.
        path (str): path to username
	# Returns
    Conn object.  Autocommit set to true to avoid locking databases.
    """
    global conn
    global tunnel
    config = read_config()

    if(tunnel):
        if(tunnel.is_active != True):
            tunnel.restart()
    else:
        tunnel = _create_tunnel(config, port)

    if(conn == None or conn.closed != 0):
        conn = psycopg2.connect(
            database=config[conn_name]['dbname'],
            user=config[conn_name]['user'],
            password = config[conn_name]['password'],
            host=tunnel.local_bind_host,
            port=tunnel.local_bind_port,
        )
        conn.autocommit = True
        return conn
    else:
        return conn

def close_connection():
    global conn
    global tunnel
    conn.close()
    tunnel.stop()
