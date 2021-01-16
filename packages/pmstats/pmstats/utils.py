from pandas.io.sql import read_sql_query as sql
from distutils.version import StrictVersion
import math
from scipy.stats import norm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pmutils import redshift



VERSION_ADOPTION= """
    SELECT
        app,
        event_date,
        COUNT(DISTINCT dw_users.user_id) as users,
        COUNT(DISTINCT CASE WHEN app_version IN ({versions}) THEN dw_users.user_id END) as updaters,
        1.0*COUNT(DISTINCT CASE WHEN app_version IN ({versions}) THEN dw_users.user_id END)/COUNT(DISTINCT dw_users.user_id) as pct_updaters
    FROM analytics.dw_user_events_daily  AS dw_daily_user_events
    JOIN analytics.dw_users  AS dw_users ON dw_daily_user_events.user_id  = dw_users.user_id
    WHERE (( COALESCE (dw_users.delete_reason, '') <> 'guest_secured') AND (dw_users.user_status <> 'restricted'))
        AND app_foreground >0 -- WEB has not version
        AND event_date BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY 1,2
    ORDER BY 1,2
"""

TOP_VERSIONS= """
    SELECT
        app,
        event_date,
        app_version,
        COUNT(DISTINCT dw_users.user_id) as users
    FROM analytics.dw_user_events_daily  AS dw_daily_user_events
    JOIN analytics.dw_users  AS dw_users ON dw_daily_user_events.user_id  = dw_users.user_id
    WHERE (( COALESCE (dw_users.delete_reason, '') <> 'guest_secured') AND (dw_users.user_status <> 'restricted'))
        AND ((app_foreground>0 AND app<>'web') OR (app='web'))
        AND event_date BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY 1,2,3
    HAVING COUNT(DISTINCT dw_users.user_id) > {min_count}
"""

TEMPLATE_DAILY = """{exposure_subquery}\n
                {metrics_subquery}\n
                SELECT\n{grouping_header}\n{metrics_list} \n
                FROM {from_table}\n{join_tables}\n{where}\n{groupby_syntax}\n{orderby_syntax}"""

TEMPLATE_USER_STATS = """{exposure_subquery}\n
                {metrics_subquery}\n
                SELECT {test_group_column},
                COUNT(1) as units,
                {metrics_stats}
                FROM
                (SELECT\n{grouping_header}\n{metrics_list} \n
                FROM {from_table}\n{join_tables}\n{where}\n{groupby_syntax}) AS users
                GROUP BY 1"""

TEMPLATE_USER_LEVEL = """{exposure_subquery}\n
                {metrics_subquery}\n
                SELECT\n{grouping_header}\n{metrics_list} \n
                FROM {from_table}\n{join_tables}\n{where}\n{groupby_syntax}
                ORDER BY RANDOM()
                LIMIT {limit_users}
                """


external_spark_sql =dict(
    event_time = """CONVERT_TIMEZONE('US/Pacific', TIMESTAMP 'epoch' + r.at * INTERVAL '1 second')""",
    desktop = """
    COALESCE(CASE
        WHEN
            lower(r.using.user_agent) LIKE '%android%' OR
            lower(r.using.user_agent) LIKE '%blackberry%' OR
            lower(r.using.user_agent) LIKE '%webos%' OR
            lower(r.using.user_agent) LIKE '%iphone%' OR
            lower(r.using.user_agent) LIKE '%iemobile%' OR
            --lower(r.using.user_agent) LIKE '%windows phone%' OR
            lower(r.using.user_agent) LIKE '%ipad%' OR
            lower(r.using.user_agent) LIKE '%ipod%' THEN 'mobile'
    ELSE
    'desktop'
    END,'None')
    """,
    browser = """
    COALESCE(CASE
            WHEN {user_agent} LIKE '%Firefox/%' THEN 'Firefox'
            WHEN {user_agent} LIKE '%Chrome/%' OR {user_agent} LIKE '%CriOS%' THEN 'Chrome'
            WHEN {user_agent} LIKE '%MSIE %' THEN 'IE'
            WHEN {user_agent} LIKE '%MSIE+%' THEN 'IE'
            WHEN {user_agent} LIKE '%Trident%' THEN 'IE'
            WHEN {user_agent} LIKE '%iPhone%' THEN 'iPhone Safari'
            WHEN {user_agent} LIKE '%iPad%' THEN 'iPad Safari'
            WHEN {user_agent} LIKE '%Opera%' THEN 'Opera'
            WHEN {user_agent} LIKE '%BlackBerry%' AND {user_agent} LIKE '%Version/%' THEN 'BlackBerry WebKit'
            WHEN {user_agent} LIKE '%BlackBerry%' THEN 'BlackBerry'
            WHEN {user_agent} LIKE '%Android%' THEN 'Android'
            WHEN {user_agent} LIKE '%Safari%' THEN 'Safari'
            WHEN {user_agent} LIKE '%bot%' THEN 'Bot'
            WHEN {user_agent} LIKE '%http://%' THEN 'Bot'
            WHEN {user_agent} LIKE '%www.%' THEN 'Bot'
            WHEN {user_agent} LIKE '%Wget%' THEN 'Bot'
            WHEN {user_agent} LIKE '%curl%' THEN 'Bot'
            WHEN {user_agent} LIKE '%urllib%' THEN 'Bot'
            ELSE 'Unknown'
        END,'None')
    """.format(user_agent="r.using.user_agent"),
    browser_version = """COALESCE(regexp_substr(r.using.user_agent, 'Firefox\/[0-9_\.]+|Chrome\/[0-9]+'),'none')""",
    os_version = """COALESCE(regexp_substr(r.using.user_agent, '(iPhone OS [0-9_\.]+|Android [0-9_\.]+|Mac OS X [0-9_\.]+|Mac OS X [0-9_\.]+|Windows NT [0-9_\.]+)'),'none')""",
    os = """COALESCE(regexp_substr(r.using.user_agent, '(iPhone OS [0-9]+|Android [0-9]+|Mac OS X [0-9]+|Mac OS X [0-9]+|Windows NT [0-9]+)'),'none')""",
)

def convert_list2comma(lst,
                name=None,
                default=None,
                quotes=True,
                defaultnull=None,
                conjunction="AND"):
    prefix = ''
    values = ''
    lst_tmp = lst
    if lst in (None, 'all'):
        lst = default

    if lst is not None:
        if not(isinstance(lst, list)):
            lst=[lst]
        if quotes is True:
            values = ", ".join(["'{0}'".format(i) for i in lst])
        else:
            values = ", ".join(["{0}".format(i) for i in lst])

    if defaultnull is not None and lst_tmp in (None,'all'):
        values = values+ ", {0}".format(defaultnull)

    if name is not None:

        prefix = "{conjunction} {name} IN ({{0}})".format(name=name, conjunction=conjunction)
        if values == '':
            output = ''
        else:
            output = prefix.format(values)
    else:
        output = values

    return output

def get_recent_versions(start_date,
                    end_date,
                    min_version,
                    query=TOP_VERSIONS,
                    min_count=100):
    top_versions = redshift(query.format(start_date=start_date, end_date=end_date, min_count=min_count))
    app_versions = top_versions['app_version'].unique()
    recent_versions=[]
    for version in app_versions:
        try:
            if StrictVersion(version) >= StrictVersion(min_version):
                recent_versions.append(version)
        except:
            pass
    return recent_versions

def get_version_adoption(start_date,
                    end_date,
                    versions,
                    query=VERSION_ADOPTION):

    return redshift(query.format(start_date=start_date,
                            end_date=end_date,
                            versions=convert_list2comma(versions)))

def ts_to_interval_hour(ts, interval):
    return pd.to_datetime(ts.date())+timedelta(hours=math.floor(ts.hour/interval)*interval)

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def get_sizing(delta_std, alpha = 0.05, power = 0.8, min_delta = 0.05, n1=14):
    normal_inv_alpha = norm.ppf((1-alpha/2))
    normal_inv_power = norm.ppf(power)
    x = math.pow(min_delta, 2)/(math.pow((normal_inv_alpha + normal_inv_power), 2) * math.pow(delta_std, 2))
    if x > 1/n1:
        days = 1/(x-1/n1)
    else:
        days = None
    return days


def get_lightspeed_sizing(control_df, test_df, **kwargs):
    """
    Take data frame with daily data for TEST and CONTROL columns to calculate estimated number days to run test.
    """
    delta = np.array(test_df)/np.array(control_df)-1
    std = np.std(delta)
    return get_sizing(std, **kwargs)

def get_outlier_stats(segment,
                    user_id='user_id',
                    grouping=['test_group','user_id'],
                    limit_users=1000000,
                    percentiles=[0.5,0.8, 0.95, 0.99, 0.999],
                    ignore_zeros=True,
                    ):
    if isinstance(segment, pd.DataFrame):
        df = segment.copy()
    else:
        df = segment.build_and_run_sql(grouping=grouping,
                                    sql_template=TEMPLATE_USER_LEVEL,
                                    limit_users=limit_users)
    if ignore_zeros is True:
        df = df.replace(0, np.nan)
    stats = df.describe(percentiles=percentiles).T
    group_stats = df.groupby(list(filter(lambda x: x!=user_id, grouping))).describe(percentiles=percentiles).T
    outlier_users = dict()
    for metric in filter(lambda x: x not in (grouping), df.columns):
        try:
            outlier_users[metric] = df[df[metric]>=stats.loc[metric, '{0}%'.format(percentiles[-1]*100.)]][grouping + [metric]]
        except Exception as e:
            print(e)


    return df, group_stats, outlier_users

def get_web_events_metrics(
                    control_segments,
                    test_segments,
                    pre_start_date,
                    pre_end_date,
                    post_start_date,
                    post_end_date,
                    segment_type='user_segment',
                    groupby=['event_type'],
                    is_page_views=False,
                    event_type=None,
                    landing_page_name=None,
                    page_name=None,
                    domain=None,
                    referrer_domain=None,
                    logged_in=None,
                    sum_column='event_count',#total_gmv
                    ):

    if groupby is None:
        groupby = "'all'"
    else:
        groupby = "||'+'||".join(["COALESCE({0},'none')".format(i) for i in groupby])

    if logged_in is None:
        logged_in=''
    else:
        logged_in='AND logged_in is {0}'.format(logged_in)

    if is_page_views:
        web_table='dw_daily_web_page_views_metrics'
    else:
        web_table='dw_daily_web_events_metrics'

    web_sql = """
        SELECT
            CASE
                WHEN {segment_type} IN ({control_segments}) THEN 'CONTROL'
                WHEN {segment_type} IN ({test_segments}) THEN 'TEST'
            ELSE 'NO TEST' END as test_group
            , event_date
            , CASE
                WHEN event_date BETWEEN '{pre_start_date}' AND '{pre_end_date}' THEN 'PRE'
                WHEN event_date BETWEEN '{post_start_date}' AND '{post_end_date}' THEN 'POST'
                ELSE 'NO TEST'
                END as test_period
            , {groupby} as event
            , sum({sum_column}) as total
        FROM analytics.{web_table}
        WHERE event_date BETWEEN '{pre_start_date}' AND '{post_end_date}'
        {event_type}
        {landing_page_name}
        {page_name}
        {domain}
        {referrer_domain}
        {logged_in}
        GROUP BY 1,2,3,4
        """.format(segment_type=segment_type,
                    control_segments=convert_list2comma(control_segments),
                    test_segments=convert_list2comma(test_segments),
                    pre_start_date=pre_start_date,
                    pre_end_date=pre_end_date,
                    post_start_date=post_start_date,
                    post_end_date=post_end_date,
                    groupby=groupby,
                    event_type=convert_list2comma(event_type,'event_type'),
                    landing_page_name=convert_list2comma(landing_page_name,'landing_page_name'),
                    page_name=convert_list2comma(page_name,'page_name'),
                    referrer_domain=convert_list2comma(referrer_domain,'referrer_domain'),
                    domain=convert_list2comma(domain,'domain'),
                    logged_in=logged_in,
                    web_table=web_table,
                    sum_column=sum_column,

                  )
    return web_sql

    
