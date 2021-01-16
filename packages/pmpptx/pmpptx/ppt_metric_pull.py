from pmutils import redshift

def pull_ppt_metrics(fields, table, months, segment='none'):

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
            order by acq_month asc
        '''.format(field_name = fields, table_name = table, num_months = months)

        metrics_df = redshift(sql_query)

    if segment != 'none':
        sql_query = '''
            select
                *
            from (
                select
                    {field_name}
                from {table_name}
                where
                    growth_segment = '{seg_names}'
                order by acq_month desc
                limit {num_months}
            ) as zz
            order by acq_month asc
        '''.format(field_name = fields, table_name = table, num_months = months, seg_names = segment)

        metrics_df = redshift(sql_query)

    return metrics_df

def pull_ppt_metrics2(fields, table, months, filter_expr = None):
    if filter_expr == None:
        sql_query = '''
            select
                *
            from (
                select
                    {field_name}
                from {table_name}
                group by acq_month
                order by acq_month desc
                limit {num_months}
            ) as zz
            order by acq_month asc
        '''.format(field_name = fields, table_name = table, num_months = months)
    else:
        sql_query = '''
            select
                *
            from (
                select
                    {field_name}
                from {table_name}
                where {filter_expr}
                group by acq_month
                order by acq_month desc
                limit {num_months}
            ) as zz
            order by acq_month asc
        '''.format(field_name = fields, table_name = table, num_months = months, filter_expr = filter_expr)
    return redshift(sql_query)
