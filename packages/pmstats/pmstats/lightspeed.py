from plotly.offline import init_notebook_mode, iplot
from .metrics import *
from .abtest import *
from .plots import *
from .utils import *
from pmutils import redshift
#from .looker_metrics import *

init_notebook_mode(connected=True)

def create_prepost(control_segments,
                   test_segments,
                   pre_start_date,
                   pre_end_date,
                   post_start_date,
                   post_end_date,
                   pre_exposure_versions=None,
                   post_exposure_versions=None,
                   segment_type='signup_segment',
                   reg_apps=None,
                   exposure_activity_apps=None,
                   metric_activity_apps=None,
                   home_domain=None,
                   standard_metrics=['ACTIVATIONS',
                                     'BUYER_ORDERS',
                                     'BUYER_OFFERS',
                                     'EVENTS',
                                     'LISTINGS',
                                    ],
                   adhoc_metrics=None,
                   adhoc_exposure=None,
                   sql_select_userids=None,
                   extra_metrics=None,
                   where=None,
                   level='daily', #daily or user_stats or user_level
                   extra_grouping=None,
                   limit_users=100000, #User Level data
                   debug_sql=False,
                   printerror = True,
                   conn=None,
                   ):

    s1 = Segments(segment_type=segment_type,
      control_segments=control_segments,
      test_segments=test_segments,
      exposure_start_date=pre_start_date,
      exposure_end_date=pre_end_date,
      reg_apps=reg_apps,
      exposure_activity_apps=exposure_activity_apps,
      metric_activity_apps=metric_activity_apps,
      exposure_versions=pre_exposure_versions,
      home_domain=home_domain,
      where=where,
      )

    s2 = Segments(segment_type=segment_type,
       control_segments=control_segments,
       test_segments=test_segments,
       exposure_start_date=post_start_date,
       exposure_end_date=post_end_date,
       reg_apps=reg_apps,
       exposure_activity_apps=exposure_activity_apps,
       metric_activity_apps=metric_activity_apps,
       exposure_versions=post_exposure_versions,
       home_domain=home_domain,
       where=where,
       )

    metrics = StandardMetricsSQL()
    exposure = adhoc_exposure or metrics.EXPOSURE
    for s in [s1, s2]:
        s.set_sql_exposure(**exposure)
        for m in standard_metrics:
            temp_m = getattr(metrics, m)
            if 'look_id' in temp_m:
                temp_m['sql'] = get_look_id_sql(look_id = temp_m['look_id'], limit='')
                temp_m['sql'] = temp_m['sql'].replace('"', "").replace(temp_m['replace'],'')
                s.add_sql_metric(name= temp_m['name'], sql = temp_m['sql'], join = temp_m['join'], metric_list=temp_m['metric_list'])
            else:
                s.add_sql_metric(**getattr(metrics, m))
        if adhoc_metrics is not None:
            for m in adhoc_metrics:
                s.add_sql_metric(**m)
        if sql_select_userids is not None:
            s.add_sql_metric(
                name="select_users",
                sql=sql_select_userids,
                join="select_users.user_id = exposure.user_id AND select_users.min_event_date <= exposure.event_date",
                metric_list=["""COUNT(DISTINCT select_users.user_id) AS cnt_select_users"""],
                how="INNER",
                is_temp_table=True,
                )

    #Generate event date
    if extra_grouping is None:
        extra_grouping = []

    if level == 'daily':
        grouping = ['test_group','event_date']
        pre = s1.build_and_run_sql(grouping=grouping + extra_grouping, sql_template=TEMPLATE_DAILY, extra_metrics=extra_metrics, debug_sql=debug_sql)
        post = s2.build_and_run_sql(grouping=grouping + extra_grouping, sql_template=TEMPLATE_DAILY, extra_metrics=extra_metrics, debug_sql=debug_sql)
        if debug_sql is False:
            pre = append_ratio_metrics(pre, metrics.LIGHTSPEED_METRICS, splitter=' per ',  printerror=printerror)
            post = append_ratio_metrics(post, metrics.LIGHTSPEED_METRICS, splitter=' per ', printerror=printerror)
    elif level == 'user_stats':
        grouping = ['test_group','user_id']
        pre = s1.build_and_run_sql(grouping=grouping + extra_grouping, sql_template=TEMPLATE_USER_STATS, extra_metrics=extra_metrics, debug_sql=debug_sql)
        post = s2.build_and_run_sql(grouping=grouping + extra_grouping, sql_template=TEMPLATE_USER_STATS, extra_metrics=extra_metrics, debug_sql=debug_sql)
    elif level == 'user_summary':
        grouping = ['test_group','user_id']
        pre = s1.build_and_run_sql(grouping=grouping + extra_grouping, sql_template=TEMPLATE_USER_LEVEL, extra_metrics=extra_metrics, limit_users=limit_users, debug_sql=debug_sql)
        post = s2.build_and_run_sql(grouping=grouping + extra_grouping, sql_template=TEMPLATE_USER_LEVEL, extra_metrics=extra_metrics, limit_users=limit_users, debug_sql=debug_sql)

    return pre, post, s1, s2

def highlight_abresult_significance(result, stats_list=['Metric',
                                      'Pre Control Metric Mean',
                                      'Pre Test Metric Mean',
                                      'Pre Delta (%)',
                                      'Post Control Metric Mean',
                                      'Post Test Metric Mean',
                                      'Post Delta (%)',
                                      'PrePost Delta (%)',
                                      'PrePost Delta LCL (%)',
                                      'PrePost Delta UCL (%)',
                                      'p-value',
                                      'Net Impact',]):

    display(result[stats_list].style.apply(highlight_significance, axis=1,diff='PrePost Delta (%)', criteria_column='p-value',).format(FORMAT_PREPOST))

def create_pretty_outputs(pre,
                        post,
                        title='AB Result',
                        max_pvalue=0.2,
                        lightspeed_only=True,
                        # TODO: show_these_metrics_only = None,
                        must_include_metrics=None,
                        must_exclude_metrics=None,
                        show_diagnostic_only=None,
                        exclude_all_lightspeed = False,
                        stats_list = ['Metric',
                                      'Pre Control Metric Mean',
                                      'Pre Test Metric Mean',
                                      'Pre Delta (%)',
                                      'Post Control Metric Mean',
                                      'Post Test Metric Mean',
                                      'Post Delta (%)',
                                      'PrePost Delta (%)',
                                      'PrePost Delta LCL (%)',
                                      'PrePost Delta UCL (%)',
                                      'p-value',
                                      'Net Impact',],
                        diagnostic=True,
                        use_plotly=True,
                        printerror=True,
                        diagnostic_max=20,
                        marker_mode='lines+markers',
                        hourly_interval=None,
                        show_tables=True,
                        xlim=None,
                        chart_xvalues=dict(mid_col='PrePost Delta (%)',
                            lo_col='PrePost Delta LCL (%)',
                            hi_col='PrePost Delta UCL (%)',),
                        is_png=False
                        ):
    metrics = StandardMetricsSQL()
    pre_copy = pre.copy()
    post_copy = post.copy()
    if hourly_interval is not None:
        for p in [pre_copy, post_copy]:
            p['event_date'] = p['event_date'].apply(lambda x: ts_to_interval_hour(x, hourly_interval))
            p = p.groupby(['test_group','event_date']).sum().reset_index()

    pre_copy = pre_copy.groupby(['test_group','event_date']).sum(min_count=1).reset_index()
    post_copy = post_copy.groupby(['test_group','event_date']).sum(min_count=1).reset_index()

    ab_result = get_prepost_stats(pre_copy, post_copy, printerror=printerror)
    filter = (ab_result['p-value'] < max_pvalue)
    if lightspeed_only:
        filter = (filter & ab_result['Metric'].isin(metrics.LIGHTSPEED_METRICS))
    if must_include_metrics is not None:
        filter = (filter|ab_result['Metric'].isin(must_include_metrics))
    if must_exclude_metrics is not None:
        filter = (filter & ~ab_result['Metric'].isin(must_exclude_metrics))
    if exclude_all_lightspeed:
        filter = (filter & ab_result['Metric'].isin(must_include_metrics))

    ab_top_result = ab_result[filter]#.sort_values(by='prepost_diff', ascending=False)
    if show_tables is True:
        display(ab_top_result[stats_list].style.apply(highlight_significance, axis=1,diff='PrePost Delta (%)', criteria_column='p-value',).format(FORMAT_PREPOST))
        print("="*100)

        fig=plotly_metrics_delta(ab_top_result, title=title, xlim=xlim, **chart_xvalues)
        if is_png is True:
            fig.show(renderer='png', width=1200)
        else:
            iplot(fig)

    diagnostic_metric = show_diagnostic_only or ab_top_result.Metric
    if diagnostic is True:
        i=0
        for metric in diagnostic_metric:
            if use_plotly is True:
                fig=plotly_prepost(pre_copy, post_copy, metric, ab_result=ab_result, marker_mode=marker_mode)
                if is_png is True:
                    fig.show(renderer='png', width=1200)
                else:
                    iplot(fig)
            else:
                plot_prepost(pre_copy, post_copy, metric)
            i+=1
            if i==diagnostic_max:
                break
    return ab_result

def _color_sig(s,
            threshold=0.05,
            threshold2=0.2,
            hi_colors=['rgb(224,102,102)', 'rgb(244,204,204)'],  #https://github.com/denilsonsa/gimp-palettes/blob/master/palettes/Google-Drive.gpl
            lo_colors=['rgb(147,196,125)', 'rgb(217, 234, 211)',]):

    if s[1] < threshold:
        if s[0] < 0:
            return 'border-color:white; background-color:{0}; font-weight:bold'.format(hi_colors[0])
        else:
            return 'border-color:white; background-color:{0}; font-weight:bold'.format(lo_colors[0])
    elif s[1]< threshold2:
        if s[0] < 0:
            return 'border-color:white; background-color:{0};'.format(hi_colors[1])
        else:
            return 'border-color:white; background-color:{0};'.format(lo_colors[1])
    else:
        return 'border-color:white'

def create_segmented_outputs(df,
        pre_start_date,
        pre_end_date,
        post_start_date,
        post_end_date,
        segmentation = None,
        values = 'num_events',
        freq='D',
        **kwargs):
    events = df.copy()
    if segmentation is None:
        segmentation = []
    if isinstance(segmentation, str):
        segmentation = [segmentation]

    events['event']= events[['event']+segmentation].apply(lambda x: "+".join(x), axis=1)
    pre, post = format_df_for_prepost(events,
        [pre_start_date, pre_end_date],
        [post_start_date, post_end_date],
        values=values,
        freq=freq,
        )

    result = create_pretty_outputs(pre,
        post,
        **kwargs
        )
    return result

def gen_summary_color_table(abresults,
                            metrics=None, #if None takes first result table Metrics
                            valcol='PrePost Delta (%)',
                            valcol2='Prepost Delta w/CI (%)',
                            sigcol='p-value',
                            display_index=0,
                            valformat="{:+.2%}"
                            ):
    try:
        i=0
        for r, result in abresults.items():
            if metrics is None:
                metrics= result['Metric']

            deltas = result.set_index('Metric')[[valcol,sigcol,valcol2]].apply(lambda row: (row[0],row[1],row[2]), axis=1).rename(r)[metrics]
            nullindex = deltas.loc[deltas.isnull()].index
            if len(nullindex)>0:
                for index in nullindex:
                    deltas[index]=[np.nan]*3

            if i==0:
                summary_deltas = deltas.to_frame()
            else:
                summary_deltas = pd.concat([summary_deltas, deltas], axis=1)
            i+=1
        display(summary_deltas.applymap(lambda x:x[display_index]).style.apply(lambda x: summary_deltas.applymap(_color_sig), axis=None).format(valformat))
    except Exception as e:
        if len(metrics) != len(set(metrics)):
            print('Err: Duplicate metrics')
        else:
            print(e)
    return summary_deltas

def format_df_for_prepost(df,
                       pre_date_range,
                       post_date_range,
                       datefield='event_date',
                       otherfields = ['test_group','segment'],
                       event='event',
                       values='num_events',
                       freq='D',
                       ):
    try:
        df[datefield]=pd.to_datetime(df[datefield])
    except:
        pass
    indexfields = [datefield]+otherfields
    pre = df.pivot_table(index=indexfields, columns=event, values=values, aggfunc=sum)
    post = df.pivot_table(index=indexfields, columns=event, values=values, aggfunc=sum)
    pre=pre[pre.index.get_level_values(datefield).isin(pd.date_range(*pre_date_range, freq=freq))].fillna(0).reset_index()
    post=post[post.index.get_level_values(datefield).isin(pd.date_range(*post_date_range, freq=freq))].fillna(0).reset_index()
    return pre, post

def filters_to_sql(filters, is_spark=True, prefix='r.', conjunction="OR", return_clause=True):
    if filters is None or filters=="":
        filters_clause = ""
        filters_agg = list()
    else:
        if isinstance(filters, dict):
            filters = [filters]

        filters_agg = list()
        if filters is not None:
            for f in filters:
                filters_line=list()
                for i,j in f.items():
                    if is_spark is False:
                        i = i.replace(".","_")
                    if isinstance(j, bool):
                        quotes=False
                    else:
                        quotes=True
                    if i=='sql':
                        filters_line.append(j)
                    else:
                        filters_line.append(convert_list2comma(j, prefix + i, conjunction="",quotes=quotes))
                filters_agg.append("({0})".format(" AND ".join(filters_line)))
            filters_clause = "({})".format("\n{} ".format(conjunction).join(filters_agg))
        else:
            filters_clause=""
    if return_clause is True:
        return filters_clause
    else:
        return filters_agg

def named_filters_to_sql(filters, is_spark=True, prefix='r.', conjunction="OR"):
    if filters is None or filters=="":
        filters_clause = ""
        case_when = ""
    else:
        filters_agg = dict()
        for name, f in filters.items():
            filters_agg[name] = filters_to_sql(f, is_spark=is_spark, prefix=prefix)

        filters_clause = "\n{} ".format(conjunction).join(filters_agg.values())
        case_when = "\n".join(["WHEN {1} THEN '{0}' ".format(i,j) for i,j in filters_agg.items()])
    return filters_clause, case_when


def get_prepost_from_events(control_segments,
    test_segments,
    pre_start_date,
    pre_end_date,
    post_start_date,
    post_end_date,
    full_date_range=False,
    actor_type='user',
    segment_type='r.actor.sign_up_segment',
    app=None,
    named_metrics=None, #dict with list of dict
    metric_filters = None, #List of dictionary of fields with their conditions, [{'verb':'view', 'direct_object.name':['closet','listing']}],
    filters=None,
    extra_where = "", #Raw SQL for additional wheres, begin with AND statement
    groupby_event=['verb'],
    conjunction='+',
    custom_groupby_event_sql = None,
    extra_segmentation=None, #dict
    values='num_events',
    is_spark=True,
    is_hour=False, #Only valid when is_spark=True
    winsorize=1, #Decimal upper bound e.g. 0.99 for 99th percentile, 1 is equivalent to max, i.e., no upper bound
    debug=False):

    event_date='r.event_date'
    freq='D'
    if is_hour is True:
        event_date = "DATE_TRUNC('hour', CONVERT_TIMEZONE('US/Pacific', TIMESTAMP 'epoch' + r.at * INTERVAL '1 second'))"
        freq = 'H'

    query_external_spark_win= """
    DROP TABLE IF EXISTS tmp_users;
    DROP TABLE IF EXISTS tmp_outlier;

    CREATE TEMP TABLE tmp_users AS
    SELECT
        {event_date} as event_date
        , r.actor.id as user_id
        , CASE
                    WHEN {segment_type} IN ({control_segments}) THEN 'CONTROL'
                    WHEN {segment_type} IN ({test_segments}) THEN 'TEST'
                ELSE 'NO TEST' END as test_group
        , {segment_type} AS segment
        , {groupby_event} as event
        {segmentation_clause}
        , COUNT(1) as num_events
        , SUM(CASE WHEN r.verb='book' THEN r.direct_object.total_price END)  as gmv
    FROM external_spark_tables.raw_events r
    WHERE ((r.event_date BETWEEN '{pre_start_date}'::DATE AND '{pre_end_date}'::DATE)
        OR (r.event_date BETWEEN '{post_start_date}'::DATE AND '{post_end_date}'::DATE))
    AND r.actor.type='{actor_type}'
    {segments}
    {app}
    {metric_filters_clause}
    {filters_clause}
    {extra_where}
    GROUP BY 1,2,3,4,5 {segmentation_indices}
    ;

    CREATE TEMP TABLE tmp_outlier AS
    SELECT
        event
        , PERCENTILE_CONT({winsorize}) WITHIN GROUP (ORDER BY num_events) as event_outlier --TODO: dynamic upper bound
    FROM tmp_users
    GROUP BY 1
    ;

    CREATE TEMP TABLE tmp_gmv_outlier AS
    SELECT
        event
        , PERCENTILE_CONT({winsorize}) WITHIN GROUP (ORDER BY gmv) as gmv_outlier --TODO: dynamic upper bound
    FROM tmp_users
    GROUP BY 1
    ;

    SELECT
        event_date
        , test_group
        , segment
        , a.event
        {segmentation_columns}
        ,COUNT(DISTINCT user_id) as users
        ,SUM(num_events) as num_events
        ,SUM(gmv) as gmv
        ,SUM(LEAST(num_events, event_outlier)) as num_events_trim
        ,SUM(LEAST(gmv, gmv_outlier)) as gmv_trim
        ,MAX(event_outlier) as event_outlier
        ,MAX(gmv_outlier) as gmv_outlier
    FROM tmp_users a
        LEFT JOIN tmp_outlier b ON a.event=b.event
        LEFT JOIN tmp_gmv_outlier c ON a.event=c.event
    GROUP BY 1,2,3,4 {segmentation_columns}
    """


    query_external_spark_simple= """
    SELECT
    {event_date} as event_date
    , CASE
                WHEN {segment_type} IN ({control_segments}) THEN 'CONTROL'
                WHEN {segment_type} IN ({test_segments}) THEN 'TEST'
            ELSE 'NO TEST' END as test_group
    , {segment_type} AS segment
    , {groupby_event} as event
    {segmentation_clause}
    , COUNT(DISTINCT r.actor.id) as users
    , COUNT(1) as num_events
    , SUM(CASE WHEN r.verb='book' THEN r.direct_object.total_price END)  as gmv
    FROM external_spark_tables.raw_events r
    WHERE ((r.event_date BETWEEN '{pre_start_date}'::DATE AND '{pre_end_date}'::DATE)
        OR (r.event_date BETWEEN '{post_start_date}'::DATE AND '{post_end_date}'::DATE))
    AND r.actor.type='{actor_type}'
    {segments}
    {app}
    {metric_filters_clause}
    {filters_clause}
    {extra_where}
    GROUP BY 1,2,3,4 {segmentation_columns}
    ORDER BY 1,2,3
    ;
    """
    if winsorize == 1:
        query_external_spark=query_external_spark_simple
    else:
        query_external_spark=query_external_spark_win

    query_dw_events_cs= """
    SELECT
    event_date
    , CASE
                WHEN {segment_type} IN ({control_segments}) THEN 'CONTROL'
                WHEN {segment_type} IN ({test_segments}) THEN 'TEST'
            ELSE 'NO TEST' END as test_group
    , {segment_type} AS segment
    , {groupby_event} as event
    {segmentation_clause}
    , SUM(count_of_distinct_events) as num_events
    FROM  analytics.dw_daily_segments_events_cs_v2
    WHERE ((event_date BETWEEN '{pre_start_date}' AND '{pre_end_date}')
        OR (event_date BETWEEN '{post_start_date}' AND '{post_end_date}'))
    {segments}
    {app}
    {metric_filters_clause}
    {filters_clause}
    {extra_where}
    GROUP BY 1,2,3,4 {segmentation_indices}
    ORDER BY 1,2,3
    ;
    """
    if is_spark is True:
        query_tmp = query_external_spark
        prefix = "r."
        app_field='r.using.app_type'
    else:
        query_tmp = query_dw_events_cs
        prefix = ""
        app_field = 'using_app_type'
        groupby_event = [i.replace('.','_') for i in groupby_event]


    if extra_segmentation is not None:
        segmentation_clause = "\n".join([", {1} AS {0}".format(i,j) for i,j in extra_segmentation.items()])
        segmentation_indices = ',' +','.join(map(str, range(5,5+len(extra_segmentation))))
        extra_segmentation = list(extra_segmentation.keys())
        segmentation_columns = ', {}'.format(', '.join(extra_segmentation))

    else:
        segmentation_clause = ''
        segmentation_indices = ''
        segmentation_columns = ''
        extra_segmentation = []

    metric_filters_clause = filters_to_sql(metric_filters, is_spark=is_spark, prefix=prefix)
    filters_clause = filters_to_sql(filters, is_spark=is_spark, prefix=prefix)

    if isinstance(groupby_event, str):
        groupby_event_sql = groupby_event
    else:
        groupby_event_sql =  "||'{0}'||".format(conjunction).join(["COALESCE({0},'none')".format(prefix + i) for i in groupby_event])
    if named_metrics is not None:
        named_metrics_clause,  named_case_when = named_filters_to_sql(named_metrics, is_spark=is_spark, prefix=prefix)
        if metric_filters is None:
            metric_filters_clause = "({0})".format(named_metrics_clause)
        else:
            metric_filters_clause = "({0} \nOR {1})".format(metric_filters_clause, named_metrics_clause)
        groupby_event_sql = """
                            CASE
                            {case_when}
                            ELSE {groupby_event_sql}
                            END
                            """.format(case_when=named_case_when,
                                        groupby_event_sql=groupby_event_sql,
                                        )
    if custom_groupby_event_sql is not None:
        groupby_event_sql = groupby_event_sql + "||'{0}'||".format(conjunction) + custom_groupby_event_sql
    if metric_filters_clause != "":
        metric_filters_clause = "AND " + metric_filters_clause


    segments=convert_list2comma(test_segments+control_segments, name=segment_type) if post_start_date is not None else ''

    if post_start_date is None:
        post_start_date=pre_start_date
        post_end_date=pre_end_date

    pre_range = [pre_start_date, pre_end_date]
    post_range = [post_start_date, post_end_date]

    if full_date_range is True:
        pre_start_date=pre_start_date
        post_start_date=pre_start_date
        pre_end_date=post_end_date
        post_end_date=post_end_date

    query=query_tmp.format(actor_type=actor_type,
            segment_type=segment_type,
            pre_start_date=pre_start_date,
            pre_end_date=pre_end_date,
            post_start_date=post_start_date,
            post_end_date=post_end_date,
            control_segments=convert_list2comma(control_segments, default=""),
            test_segments=convert_list2comma(test_segments, default=""),
            segments=segments,
            app=convert_list2comma(app, app_field, default=['web', 'iphone','android','ipad']),
            groupby_event=groupby_event_sql,
            metric_filters_clause=metric_filters_clause,
            filters_clause=filters_clause,
            extra_where=extra_where,
            event_date=event_date,
            segmentation_clause=segmentation_clause,
            segmentation_indices=segmentation_indices,
            segmentation_columns=segmentation_columns,
            winsorize=winsorize,
            )
    if debug is True:
        events=None
        pre_events=None
        post_events=None
        print(pre_range, post_range, values,freq)
    else:
        events=redshift(query)
        if isinstance(values, str):
            pre_events, post_events = format_df_for_prepost(events,
                                        pre_range,
                                        post_range,
                                        otherfields=['test_group','segment']+extra_segmentation,
                                        values=values,
                                        freq=freq)
        else:
            pre_events = dict()
            post_events = dict()
            for v in values:
                try:
                    pre_events[v], post_events[v] = format_df_for_prepost(events,
                                            pre_range,
                                            post_range,
                                            otherfields=['test_group','segment']+extra_segmentation,
                                            values=v,
                                            freq=freq)
                except:
                    pass

    return pre_events, post_events, events, query


def format_test_dates(post_start_date,
                         post_end_date,
                         pre_start_date=None,
                         pre_end_date=None,
                         is_hour=False,
                         adjust_post=False):

    if is_hour is True:
        return dict(pre_start_date=pd.to_datetime(pre_start_date),
            pre_end_date=pd.to_datetime(pre_end_date+' 23:59:59'),
            post_start_date=pd.Timestamp(post_start_date).round('1h'),
            post_end_date=pd.Timestamp(post_end_date).round('1h'),
                                 )
    else:
        if adjust_post is True:
            post_start_date=pd.to_datetime(post_start_date).date()+pd.to_timedelta(1, unit='d')
            post_end_date=pd.to_datetime(post_end_date).date()-pd.to_timedelta(1, unit='d')

        return dict(pre_start_date=pre_start_date,
            pre_end_date=pre_end_date,
            post_start_date=post_start_date,
            post_end_date=post_end_date,
            )

def get_segment_bias(pre, post, control_segments, test_segments, **kwargs):
    bias=dict()
    for c in control_segments:
        for t in test_segments:
            new_pre=pre[pre['segment'].isin([c,t])]
            new_post=post[post['segment'].isin([c,t])]
            new_pre['test_group']=new_pre['segment'].apply(lambda x: 'TEST' if x==t else 'CONTROL')
            new_post['test_group']=new_post['segment'].apply(lambda x: 'TEST' if x==t else 'CONTROL')
            bias['{0}{1}'.format(c,t)]=create_pretty_outputs(
                new_pre,
                new_post,
                title='AB Test - Bias c={0}, t={1}'.format(c,t),
                **kwargs
                )

    return bias
