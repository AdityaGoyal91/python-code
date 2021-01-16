from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from pmutils.creds import read_config, _create_tunnel
import configparser
import sqlalchemy
import keyring
import getpass
import pmutils
import random
import pandas

def clean_query(query=None):
    return query.replace(r'%%','*****').replace('%',r'%%').replace('*****',r'%%')

def redshift(query='select 1 as "test"',
             config=read_config(),
             section='redshift',
             execute=False,
             port=random.choice(range(3000,7000))):

    # check for file-based queries and escape % characters.
    if query.endswith('.sql'):
        with open(query,'r') as q:
            query = q.read()
    query = clean_query(query)

    # Create an SSH tunnel
    with SSHTunnelForwarder(
        (config.get(section,'tunnel_host'), config.getint(section,'tunnel_port')),
        ssh_username=config.get(section,'user'),
        remote_bind_address=(config.get(section,'host'), config.getint(section,'port')),
        local_bind_address=('localhost',port), # could be any available port
        ) as tunnel:
        tunnel.start()
        engine_type = 'postgres'
        user=config.get(section,'user')
        password = config.get(section,'password')
        host=tunnel.local_bind_host
        port=tunnel.local_bind_port
        database=config.get(section, 'dbname')
        jdbc = f'{engine_type}://{user}:{password}@{host}:{port}/{database}'
        with sqlalchemy.create_engine(jdbc, poolclass=NullPool, isolation_level='AUTOCOMMIT'
        ).connect() as connection:
            conn = connection.execution_options(autocommit=True)
            if execute:
                conn.execute(query)
            else:
                return pandas.read_sql(query, conn)

def redshift_engine(
             config=read_config(),
             section='redshift',
             get_jdbc=False,
             port=random.choice(range(3000,7000))):

    global tunnel
    config = read_config()
    try:
        if(tunnel):
            if(tunnel.is_active != True):
                tunnel.start()
            else:
                tunnel = _create_tunnel(config, port)
    except:
        tunnel = _create_tunnel(config, port)

    engine_type = 'postgres'
    user=config.get(section,'user')
    password = config.get(section,'password')
    host=tunnel.local_bind_host
    port=tunnel.local_bind_port
    database=config.get(section, 'dbname')
    if get_jdbc:
        return f'{engine_type}://{user}:{password}@{host}:{port}/{database}'.replace(password, '{password}')
    else:
        jdbc = f'{engine_type}://{user}:{password}@{host}:{port}/{database}'
        return {'engine':sqlalchemy.create_engine(jdbc, poolclass=NullPool, isolation_level='AUTOCOMMIT'),
            'tunnel':tunnel}


def redshift_stage(query='select 1 as "test"',
                   config=read_config(),
                   section='redshift_stage',
                   engine=False,
                   port=random.choice(range(3000,7000))):
    # Create an SSH tunnel
    return 'redshift_stage is still in development. Try again at a future date.'
    engine_type = 'postgres'
    user=config.get(section,'user')
    password = config.get(section,'password')
    host=config.get(section,'host')
    port=config.get(section,'port')
    database=config.get(section, 'dbname')
    jdbc = f'{engine_type}://{user}:{password}@{host}:{port}/{database}'
    if engine:
        return sqlalchemy.create_engine(jdbc, poolclass=NullPool, isolation_level='AUTOCOMMIT')
    else:
        with sqlalchemy.create_engine(jdbc, poolclass=NullPool, isolation_level='AUTOCOMMIT').connect() as conn:
            if query.endswith('.sql'):
                with open(query,'r') as q:
                    return pandas.read_sql(q.read(), conn)
            else:
                return pandas.read_sql(query, conn)
