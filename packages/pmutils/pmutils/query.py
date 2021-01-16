from pandas.io.sql import read_sql_query as sql
from pmutils.redshift import redshift



def run_sql(query, conn):
    if '.sql' == query[-4:]:
        file=open(query, 'r')
        q=file.read()
        file.close()
        return sql(q, conn)
    else:
        return sql(query, conn)


# uses an existing connection to run a sql query from a string or a file.
# if a connection isn't specified a new connection (including ssh tunnel)
# is created for the duration of the query.
def run_sql_alt(query='select 1 as "test"', conn=None):

    if query.endswith('.sql'):
        with open(query, 'r') as f:
            query = f.read()

    if conn:
        return sql(query, conn)
    else:
        return redshift(query)
