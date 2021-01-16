"""
AB Testing data pull and stats calculations
"""
from pandas.io.sql import read_sql_query as sql
from collections import OrderedDict
import pandas as pd
import statsmodels.stats.api as sms
from  statsmodels.stats.power import tt_ind_solve_power
from statsmodels.sandbox.stats.runs import runstest_1samp
import statsmodels.stats.weightstats as wstats
import numpy as np
from scipy import stats
import re
from .utils import convert_list2comma
from pmutils import redshift

idx = pd.IndexSlice

def test():
    print("Yay! You got the import to work, chk")

class Segments:
    """
    Class with filters and metrics for segments in Test and Control.
    # Arguments
        segment_type (str): signup_segment or sign_up_browser_segment from dw_users
        control_segments (list of str): List of segment letters for control group
        test_segments (list of str): List of segment letters for test group
        exposure_version (list of str): List of app version to filter by for exposure group, default None does not filter any app version.
        exposure_start_date (yyyy-mm-dd): Start date of exposure. Required.
        exposure_end_date (yyyy-mm-dd): End date of exposure. Required.
        metric_start_date (yyyy-mm-dd): Start date of all metrics.  None defaults to exposure start date.
        metric_end_date (yyyy-mm-dd): End of all metrics.  None defaults to exposure end date.
        sql_exposure (str): SQL of exposure group
        sql_metric(list of dictionary): List of dictionary for SQL of all metrics.
        reg_apps (list of str): List of reg apps to filter by, default None does not filter any reg app
        exposure_activity_apps (list of str): List of apps to filter by for exposure group, default None does not filter any apps
        metric_activity_apps (list of str): List of apps to filter by for metric group, default None does not filter any apps
        where (str): Complete where statement that is added to the final built query. Default is ''.
    # Attributes
        sql: The sql that is built from the parameters provided.
    """
    def __init__(self,
                 conn=None, #Deprecating
                 segment_type='signup_segment', #sign_up_browser_segment
                 control_segments=['A'],
                 test_segments=['B'],
                 exclude_pre_exposure=True,
                 exposure_versions=None,
                 exposure_start_date='2018-11-01',
                 exposure_end_date='2018-11-02',
                 metric_start_date=None,
                 metric_end_date=None,
                 sql_exposure=None,
                 sql_metric=None,
                 reg_apps=None,
                 home_domain=None,
                 exposure_activity_apps=None,
                 metric_activity_apps=None,
                 where=None,
                 ):

        self.filters = {}
        self.filters['segment_type'] = segment_type
        self.filters['segments'] = ", ".join(["'{0}'".format(i) for i in control_segments + test_segments])
        self.filters['control_segments'] = ", ".join(["'{0}'".format(i) for i in control_segments])
        self.filters['test_segments'] = ", ".join(["'{0}'".format(i) for i in test_segments])
        self.num_test_segments = len(test_segments)

        if exclude_pre_exposure==True:
            self.filters['exclude_pre_exposure'] = 'HAVING d_dates.full_dt >= MIN(dw_daily_user_events.event_date)'
        else:
            self.filters['exclude_pre_exposure'] = ''

        self.filters['exposure_start_date'], self.filters['exposure_end_date'] = exposure_start_date, exposure_end_date
        self.filters['metric_start_date'] = metric_start_date or exposure_start_date
        self.filters['metric_end_date'] = metric_end_date or exposure_end_date
        self.sql_exposure = sql_exposure
        self.sql_metric = {}
        if sql_metric != None:
            self.sql_metric = sql_metric




        if exposure_versions in (None, 'all') :
            self.filters['exposure_versions'] = ''
        else:
            self.filters['exposure_versions'] = "AND (app='web' OR (app<> 'web' AND app_version IN ({0})))".format(", ".join(["'{0}'".format(i) for i in exposure_versions])) #Web doesn't have versions

        self.filters['reg_apps'] = convert_list2comma(reg_apps, default=['android', 'iphone', 'ipad', 'web'], quotes=True, defaultnull='NULL')
        self.filters['exposure_activity_apps'] = convert_list2comma(exposure_activity_apps, default=['android', 'iphone', 'ipad', 'web'], quotes=True, defaultnull='NULL')
        self.filters['metric_activity_apps'] = convert_list2comma(metric_activity_apps, default=['android', 'iphone', 'ipad', 'web'], quotes=True, defaultnull='NULL')
        self.filters['home_domain'] =  convert_list2comma(home_domain, default=['ca', 'us'], quotes=True)
        self.where=where

    def set_sql_exposure(self, sql, name='exposure', metric_list=None, **kwargs):
        """SQL for exposure with paramters for .format().
        # Arguments
            sql (str): SQL of exposure with .format() parameters. TODO: sql
            name (str): Name of alias used in query.
            kwargs (str): Other parameters used in .format()
        """

        filters = self.filters
        for i, j in kwargs.items():
            filters[i] = j

        self.sql_exposure = [name, sql.format(**filters), metric_list]

    def add_sql_metric(self, name, sql, join, metric_list=None, how='LEFT', is_temp_table=False, **kwargs):
        """SQL for a metric with paramters for .format().
        # Arguments
            sql (str): SQL of exposure with .format() parameters.
            name (str): Name of subquery alias. TODO: alias
            join (str): Join statement with exposure subquery
            metric_list (list of str): List of metrics sql to pull to main table.
            kwargs (str): Other parameters used in .format()
        """
        filters = self.filters

        temp_table = None

        if is_temp_table is True:
            temp_table="""
            DROP TABLE IF EXISTS tmp_{name};
            CREATE TEMP TABLE tmp_{name} AS
            {sql}
            """.format(name=name, sql=sql)

            sql="SELECT * FROM tmp_{name}".format(name=name) #OVERWRITE SQL W GENERIC SELECT *

        for i, j in kwargs.items():
            filters[i] = j
        self.sql_metric[name] = [sql.format(**filters), join, metric_list, how, temp_table]

    def build_and_run_sql(self,
                  grouping=['test_group', 'event_date'],
                  extra_metrics=None,
                  where=None,
                  sql_template="""{exposure_subquery}\n
                                  {metrics_subquery}\n
                                  SELECT\n{grouping_header}\n{metrics_list} \n
                                  FROM {from_table}\n{join_tables}\n{where}\n{groupby_syntax}\n{orderby_syntax}""",
                  test_group_column='test_group',
                  debug_sql=False,
                  **kwargs):
        """Build and run sql based exposure and metrics SQL
        # Arguments
            grouping (list of str): Main grouping of the data, should include test_group at the minimum.
            extra_metrics (list of str): List of additional metrics sql to pull to main table.
            where (str): String with WHERE portion of final SQL.
        # Returns
            Dataframe from SQL built from the exposure and metrics subqueries.
        """

        exposure_subquery = """WITH {0} AS ({1}),\n""".format(*self.sql_exposure)
        metrics_subquery = "\n, ".join(["{0} AS ({1})".format(i, j[0]) for i, j in self.sql_metric.items()])
        #print(tmp_table_subquery)
        metrics_calc = [metric for s, j, m, n, t in self.sql_metric.values() for metric in m]+(extra_metrics or [])
        metrics_list = ", " + "\n, ".join(metrics_calc)
        self.metrics_names = [re.findall(r'(?<=as ).+',i, re.IGNORECASE)[-1].strip().strip('\"') for i in metrics_calc]
        metrics_stats = ", " .join(["""AVG(COALESCE("{0}",0)::FLOAT) AS "{0} MEAN",
                                       STDDEV(1.0*COALESCE("{0}",0)) AS "{0} STD",
                                       MAX(1.0*COALESCE("{0}",0)) AS "{0} MAX"
                                    """.format(i) for i in self.metrics_names]) # necessary for user level calculations. Doesn't work well for ratios and null values.
        join_tables = '\n'.join(["{0} JOIN {1} ON {2}".format(j[3], i, j[1]) for i, j in self.sql_metric.items()])
        if isinstance(grouping, str):
            grouping = [grouping]
        grouping_header = ",\n".join(["{0}.{1}".format(self.sql_exposure[0], g) for g in grouping])
        groupby_syntax = "GROUP BY "+", ".join([str(i) for i in range(1, len(grouping)+1)])
        orderby_syntax = "ORDER BY "+", ".join([str(i) for i in range(1, len(grouping)+1)])
        final_sql = sql_template.format(exposure_subquery=exposure_subquery,
                   metrics_subquery=metrics_subquery,
                   grouping_header=grouping_header,
                   metrics_list=metrics_list,
                   from_table=self.sql_exposure[0],
                   join_tables=join_tables,
                   where=where or self.where or '',
                   groupby_syntax=groupby_syntax,
                   orderby_syntax=orderby_syntax,
                   test_group_column=test_group_column, #User level testing
                   metrics_stats=metrics_stats, #User level testing
                   **kwargs
                  )
        tmp_table_list = [j[4] for i, j in self.sql_metric.items() if j[4] is not None]

        if len(tmp_table_list)>0:
            final_sql = ";\n".join(tmp_table_list) + ";\n" + final_sql

        self.sql = final_sql.format(**self.filters)
        if debug_sql is True:
            return self.sql
        else:
            return redshift(self.sql)

    def build_and_run_sql_iter(self,  #NOT FINISHED
                  grouping=['test_group', 'event_date'],
                  where=None,
                  sql_template="""WITH {from_table} AS ({exposure_sql})
                                    , {metric} AS ({metric_sql})
                                    SELECT\n{grouping_header}\n{metrics_list}
                                    FROM {from_table}
                                    {how} JOIN {metric}
                                    ON {join}
                                    {where}\n{groupby_syntax}""",
                  test_group_column='test_group',
                  **kwargs):
        """Build and run sql based exposure and metrics SQL
        # Arguments
            grouping (list of str): Main grouping of the data, should include test_group at the minimum.
            extra_metrics (list of str): List of additional metrics sql to pull to main table.
            where (str): String with WHERE portion of final SQL.
        # Returns
            Dataframe from SQL built from the exposure and metrics subqueries.
        """
        sql_params = dict()
        sql_params['from_table']= self.sql_exposure[0]
        sql_params['exposure_sql']= self.sql_exposure[1]
        sql_params['grouping_header'] = ",\n".join(["{0}.{1}".format(sql_params['from_table'], g) for g in grouping])
        sql_params['groupby_syntax'] = "GROUP BY "+", ".join([str(i) for i in range(1, len(grouping)+1)]) + "\nORDER BY "+", ".join([str(i) for i in range(1, len(grouping)+1)])
        sql_params['where']=''

        df_list=dict()
        for m, q in self.sql_metric.items():
            sql_params.update(**dict(zip(['metric','metric_sql', 'join','metrics_list','how'], [m]+q)))
            sql_params['metrics_list'] = ", " + "\n, ".join(sql_params['metrics_list'])
            df_list[m] = redshift(sql_template.format(**sql_params).format(**self.filters))
        return df_list



def get_prepost_stats(pre,
                      post,
                      test_group_column='test_group',
                      experiment_unit='event_date',
                      control='CONTROL',
                      test='TEST',
                      alpha=0.05,
                      printerror=True):
    """Generate Pre-Post statistics given 2 dataframes from Pre and Post periods with test and control groups.
    # Arguments
        pre (DataFrame): Pandas dataframe for pre data, must include control and test groups.
        post (DataFrame): Pandas dataframe for post data, must include control and test groups.
        experiment_unit (str): Experiment unit for stats.
        test_group_column: Column used to identify test and control groups.
        control (str): Name of control group in test_group_column. Default: CONTROL.
        test (str): Name of test group in test_group_column. Default: TEST.
        alpha (float): Significance level for calculating p-value and confidence intervals.
    # Returns
        Dataframe for each metric with pre-post summary and statistics.
    """

    df = {}
    df['pre'] = pre
    df['post'] = post
    metrics = df['pre'].drop([test_group_column, experiment_unit], axis=1).columns
    results = []

    for metric in metrics:
        # if (pre[metric].count()<=2)|(pre[metric].count()<=1):
        #     print('Insufficient data: '+ metric)
        #     continue #Skip empty results
        stats = {'pre':{},
                 'post':{}}

        try:
            for i in ('pre', 'post'):
                stats[i]=get_relative_diff(df[i], i, metric)['stats']
            cm = sms.CompareMeans(stats['post']['desc'], stats['pre']['desc'])
            ci = cm.tconfint_diff(usevar='unequal')
            t, p, dof = cm.ttest_ind(usevar='unequal')
            power = tt_ind_solve_power(effect_size=t, nobs1=stats['pre']['nobs'], ratio=stats['post']['nobs']/stats['pre']['nobs'], alpha=0.05)
        except Exception as e:
            if printerror is True:
                print(e)
                #print('Insufficient data: '+ metric)
            continue #Skip empty results

        results.append(
            OrderedDict(
                metric=metric,
                pre_days=stats['pre']['nobs'],
                pre_control_metric_sum=stats['pre']['control_metric_sum'],
                pre_test_metric_sum=stats['pre']['test_metric_sum'],
                pre_control_metric_mean=stats['pre']['control_metric_mean'],
                pre_test_metric_mean=stats['pre']['test_metric_mean'],
                pre_delta_mean=stats['pre']['mean'],
                pre_delta_lcl=stats['pre']['metric_delta_lcl'],
                pre_delta_ucl=stats['pre']['metric_delta_ucl'],
                post_days=stats['post']['nobs'],
                post_control_metric_sum=stats['post']['control_metric_sum'],
                post_test_metric_sum=stats['post']['test_metric_sum'],
                post_control_metric_mean=stats['post']['control_metric_mean'],
                post_test_metric_mean=stats['post']['test_metric_mean'],
                post_delta_mean=stats['post']['mean'],
                post_delta_lcl=stats['post']['metric_delta_lcl'],
                post_delta_ucl=stats['post']['metric_delta_ucl'],
                prepost_delta=stats['post']['mean']-stats['pre']['mean'],
                prepost_delta_lcl=ci[0],
                prepost_delta_ucl=ci[1],
                prepost_delta_plus_minus=(ci[1]-ci[0])/2,

                prepost_delta_pvalue=p,
                net_impact=stats['post']['control_metric_mean']*(stats['post']['mean']-stats['pre']['mean']),
                net_lcl=stats['post']['control_metric_mean']*(1+ci[0]),
                net_ucl=stats['post']['control_metric_mean']*(1+ci[1]),
                net_plus_minus=stats['post']['control_metric_mean']*(ci[1]-ci[0])/2,
                power=power,
            ))

    results_df = pd.DataFrame(results)
    try:
        results_df['Prepost Delta w/CI (%)']=results_df[['prepost_delta','prepost_delta_plus_minus']].apply(lambda row: '{0:+.2f}\u00B1{1:.2f}%'.format(*row*100) if not(pd.isnull(row[0])) else '-' , axis=1)
    except:
        results_df['Prepost Delta w/CI (%)']='-'

    results_df.rename(columns=dict(
                metric='Metric',
                pre_days='Pre Days',
                pre_control_metric_sum="Pre Control Metric Sum",
                pre_test_metric_sum="Pre Test Metric Sum",
                pre_control_metric_mean="Pre Control Metric Mean",
                pre_test_metric_mean="Pre Test Metric Mean",
                pre_delta_mean="Pre Delta (%)",
                pre_delta_lcl="Pre Delta LCL (%)",
                pre_delta_ucl="Pre Delta UCL (%)",
                post_days="Post Days",
                post_control_metric_sum="Post Control Metric Sum",
                post_test_metric_sum="Post Test Metric Sum",
                post_control_metric_mean="Post Control Metric Mean",
                post_test_metric_mean="Post Test Metric Mean",
                post_delta_mean="Post Delta (%)",
                post_delta_lcl="Post Delta LCL (%)",
                post_delta_ucl="Post Delta UCL (%)",
                prepost_delta="PrePost Delta (%)",
                prepost_delta_lcl="PrePost Delta LCL (%)",
                prepost_delta_ucl="PrePost Delta UCL (%)",
                prepost_delta_pvalue="p-value",
                net_impact="Net Impact",
                power='Power (%)'
                    )
                , inplace=True)

    #Extra calculations
    return results_df


def get_relative_diff(df,
                      name,
                      metric,
                      test_group_column='test_group',
                      experiment_unit='event_date',
                      control='CONTROL',
                      test='TEST'):
    "Returns dataframe with relative differences between test and control"

    summary_df = df.set_index([test_group_column, experiment_unit]).unstack(test_group_column)
    control_df=summary_df.loc[:, idx[metric, control]]
    test_df=summary_df.loc[:, idx[metric, test]]
    control_df = control_df[control_df.notnull()]
    test_df = test_df[test_df.notnull()]
    result = pd.concat([control_df, test_df, test_df/control_df -1,], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    result['name'] = name
    result.columns  = ['control', 'test', 'rel_diff', 'name']
    stats = dict()
    stats['control_metric_sum'] = control_df.sum()
    stats['test_metric_sum'] = test_df.sum()
    stats['control_metric_mean'] = control_df.mean()
    stats['test_metric_mean'] = test_df.mean()
    stats['desc'] = sms.DescrStatsW(result['rel_diff'])
    stats['nobs'] = stats['desc'].nobs
    stats['mean'] = stats['desc'].mean
    stats['metric_delta_lcl'], stats['metric_delta_ucl'] = stats['desc'].tconfint_mean()
    stats['runs_test'] = runstest_1samp(result['rel_diff'], cutoff='mean')
    return dict(data=result, stats=stats)


def get_peruser_stats(df,
                      metrics,
                      mean_suffix='mean',
                      std_suffix='std',
                      test_group_column='test_group',
                      control='CONTROL',
                      test='TEST',
                      alpha=0.05):
    stats = list()
    for metric in metrics:
        test_mean, control_mean, diff, confint, zstat, pvalue = get_peruser_diff_zstat(df,
                            'units',
                            '{0} {1}'.format(metric, mean_suffix),
                            '{0} {1}'.format(metric, std_suffix),
                            test_group_column=test_group_column,
                            test=test,
                            control=control,
                            alpha=alpha,
                            )
        stats.append(OrderedDict(
                        metric=metric,
                        test_mean=test_mean,
                        control_mean=control_mean,
                        diff=diff,
                        lcl=confint[0],
                        ucl=confint[1],
                        zstat=zstat,
                        pvalue=pvalue,
                        ))


    return pd.DataFrame(stats)

def get_peruser_diff_zstat(df,
                  nobs_name,
                  mean_name,
                  std_name,
                  test_group_column='test_group',
                  test='TEST',
                  control='CONTROL',
                  alpha=0.05,
                  alternative='two-sided'):
        '''confidence interval for the difference in means.
        Similar to https://www.statsmodels.org/dev/generated/statsmodels.stats.weightstats.CompareMeans.zconfint_diff.html#statsmodels.stats.weightstats.CompareMeans.zconfint_diff
        Example: pre = s1.build_and_run_sql(grouping=['test_group','user_id'], sql_template=TEMPLATE_USER_STATS)
        get_user_agg_zstat(pre, 'units','buyers mean', 'buyers std')
        Parameters
        ----------
        alpha : float
            significance level for the confidence interval, coverage is
            ``1-alpha``
        alternative : string
            This specifies the alternative hypothesis for the test that
            corresponds to the confidence interval.
            The alternative hypothesis, H1, has to be one of the following :
            'two-sided': H1: difference in means not equal to value (default)
            'larger' :   H1: difference in means larger than value
            'smaller' :  H1: difference in means smaller than value
        Returns
        -------
        diff, zstat, pvalue, confint : floats
        '''
        summary = df.set_index(test_group_column)#.T.loc[[nobs_name, mean_name, std_name],[test, control]].to_dict()

        d1 = summary.loc[test, :]
        d2 = summary.loc[control,:]
        diff = d1[mean_name] - d2[mean_name]
        std_diff = np.sqrt(d1[std_name]**2/(d1[nobs_name]-1) + d2[std_name]**2/(d2[nobs_name]-1)) #Assume unequal variance
        confint = wstats._zconfint_generic(diff, std_diff, alpha=alpha, alternative=alternative)
        zstat, pvalue = wstats._zstat_generic2(diff, std_diff, alternative)
        return d1[mean_name], d2[mean_name], diff/d2[mean_name], confint/d2[mean_name], zstat, pvalue

def append_ratio_metrics(df, metrics, splitter= ' per ', printerror = True):
    """Take a list of ratio metrics and creates them for the assigned dataframe.
    Example: 'gmv/dau' will be converted to gmv per dau"""
    df_tmp = df.copy()
    for m in metrics:
        ratio = m.split(splitter)
        if len(ratio) == 2:
            try:
                df_tmp["{0} per {1}".format(*ratio)] = df_tmp[ratio[0]]/df_tmp[ratio[1]]
            except:
                if printerror:
                    print('Ratio Error: '+ m)
    return df_tmp

def highlight_significance(s,
                            diff,
                            criteria_column,
                            threshold=0.05,
                            threshold2=0.2,
                            hi_colors=['rgb(204,0,0)', 'salmon'],  #https://github.com/denilsonsa/gimp-palettes/blob/master/palettes/Google-Drive.gpl
                            lo_colors=['rgb(106,168,79)', 'rgb(147,196,125)',]):
    "Highlight dataframe based on threshold on the criteria column"
    if s[criteria_column]< threshold:
        if s[diff] < 0:
            return ['color:{0}; font-weight:bold'.format(hi_colors[0])]*len(s)
        else:
            return ['color:{0}; font-weight:bold'.format(lo_colors[0])]*len(s)
    elif s[criteria_column]< threshold2:
        if s[diff] < 0:
            return ['color:{0};'.format(hi_colors[1])]*len(s)
        else:
            return ['color:{0};'.format(lo_colors[1])]*len(s) #keeping both in case want to make it different
    else:
        return ['']*len(s)

def transform_prepost_row2col(df,
                test_group='test_group',
                event_date='event_date',
                event='event',
                value='num_events'):
    df_tmp = df.copy()
    df_tmp = df_tmp[[test_group, event_date, event, value]].set_index([test_group, event_date, event]).unstack(event)
    df_tmp.columns = df_tmp.columns.droplevel()
    return df_tmp


FORMAT_PREPOST = {
                "Pre Control Metric Sum":'{:,.2f}',
                "Pre Test Metric Sum":'{:,.2f}',
                "Pre Control Metric Mean":'{:,.2f}',
                "Pre Test Metric Mean":'{:,.2f}',
                "Pre Delta (%)":'{:,.2%}',
                "Pre Delta LCL (%)":'{:,.2%}',
                "Pre Delta UCL (%)":'{:,.2%}',
                "Post Control Metric Sum":'{:,.2f}',
                "Post Test Metric Sum":'{:,.2f}',
                "Post Control Metric Mean":'{:,.2f}',
                "Post Test Metric Mean":'{:,.2f}',
                "Post Delta (%)":'{:,.2%}',
                "Post Delta LCL (%)":'{:,.2%}',
                "Post Delta UCL (%)":'{:,.2%}',
                "PrePost Delta (%)":'{:,.2%}',
                "PrePost Delta LCL (%)":'{:,.2%}',
                "PrePost Delta UCL (%)":'{:,.2%}',
                "p-value":'{:,.4f}',
                "Net Daily Impact":'{:,.3f}',
                "Power (%)":'{:,.2%}',
                }
