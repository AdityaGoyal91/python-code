from pmutils.redshift import redshift
from pmstats import utils
import pandas as pd
import numpy as np

def get_events(date, app):
    all_views_query = """
        SELECT v.direct_object_name,
            SUM(count_of_distinct_events) as total_views
        FROM analytics.dw_daily_events_cs as v
        WHERE v.event_date > {date}
            AND using_app_type in ({app})
            AND verb = 'view'
        GROUP BY 1""".format(date = date, app = utils.convert_list2comma(app))

    all_clicks_query = """
        SELECT v.on_name,
            v.direct_object_name,
            SUM(count_of_distinct_events) as total_clicks
        FROM analytics.dw_daily_events_cs as v
        WHERE v.event_date > {date}
            AND using_app_type in ({app})
            AND verb = 'click'
        GROUP BY 1, 2;""".format(date = date, app = utils.convert_list2comma(app))

    all_views = redshift(all_views_query)
    all_clicks = redshift(all_clicks_query)
    return all_views, all_clicks
    # events = {}
    # missing_views = []

    # for i in all_views.direct_object_name:
    #     events[i] = []

    # for index, row in all_clicks.iterrows():
    #     try:
    #         events[row.on_name].append(row.direct_object_name)
    #     except KeyError:
    #         missing_views.append((row.on_name, row.direct_object_name))

    # return events, missing_views
