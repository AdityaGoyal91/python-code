import pandas.io.sql as pdsql
import psycopg2

def pull_ppt_metrics(fields, table, months, conn, segment='none'):

    if segment == 'none':
        sql_query = '''
            select
                *
            from (
                select
                    {field_name}
                from {table_name}
                order by acq_month desc
                limit {num_months}
            ) as zz
            order by month asc
        '''.format(field_name = fields, table_name = table, num_months = months)

        metrics_df = pdsql.read_sql_query(sql_query, con = conn)

    if segment != 'none':
        sql_query = '''
            select
                *
            from (
                select
                    {field_name}
                from {table_name}
                where growth_segment = '{seg_names}'
                order by acq_month desc
                limit {num_months}
            ) as zz
            order by month asc
        '''.format(field_name = fields, table_name = table, num_months = months, seg_names = segment)

        metrics_df = pdsql.read_sql_query(sql_query, con = conn)

    return metrics_df
