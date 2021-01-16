"""
Assignment algorithm for test segments
"""

from itertools import combinations
import math
from numpy.random import shuffle
import pandas as pd
import numpy as np
from pmutils import redshift
import plotly.figure_factory as ff


def get_test_units(tests, ascending=False):
    segtests = tests.astype(str).agg(''.join, axis=0)
    inv_segtests = (-1*tests).astype(str).agg(''.join, axis=0)
    test_pairs = pd.concat([segtests, inv_segtests], axis=1)
    test_pairs['pairs']=test_pairs.apply(lambda r: sorted(r), axis=1)
    test_pairs['p0']=test_pairs['pairs'].apply(lambda r: r[0])
    test_pairs['p1']=test_pairs['pairs'].apply(lambda r: r[1])
    test_pairs['level'] = (test_pairs['p0']==test_pairs[0]).astype(int)
    test_pairs['is_empty']=(test_pairs['p0']==test_pairs['p1']).astype(int)
    pair_size = test_pairs.groupby('p0').size().sort_values(ascending=ascending).reset_index()
    test_pairs=test_pairs.reset_index()
    test_units = dict()
    for i, p in enumerate(pair_size['p0']):
        pairs = test_pairs[test_pairs.p0==p][['index', 0,'p0','level', 'is_empty']]
        try:
            if int(p)==0:
                i=99 #Just dummy value for identifying empty segments.
        except:
            pass
        test_units[i]=pairs
    return test_units, test_pairs

def assign_test_unit_index(a, n_test=2, random=True, is_pair=True, variants=1):
    """
    Example:
    a = [1]*8+[0]*16+[-1]*8
    b = [0]*12+[-1]*4+[1]*4+[0]*12
    c = [1]*16+[-1]*16
    tests = pd.DataFrame(np.array([a,b]), columns=range(32))
    assignments = assign_segments(tests, n_test=2, empty_first=True)
    """
    excess=0
    if variants==1:
        # Subset array to t0 and t1
        if is_pair is True:
            segments = dict(t0 = a[a.level==0]['index'],
                            t1 = a[a.level==1]['index'],)
            # For example test size 8--> 1000, which has length 4. 2^(4-1)=8 max test size.
            max_test = int(2**(len(format(len(segments['t0']),'b'))-1))
        else:
            max_test = math.floor(len(a['index'])/2)
            # comb = np.array(list(combinations(a['index'], max_test)))
            # if random is True:
            #     shuffle(comb)
            # segments = dict(t0 = a[a.index.isin(comb[0])]['index'],
            #                 t1 = a[~a.index.isin(comb[0])]['index'], )

        """
        Using binary conversion estimate maximum size of test that can overlap.
        Test size overlap are limited to powers of 2 and must be minimum size 2.
        So only test group number of segment sizes are practically
        limited to 2,4,6,8,16.
        """

        if n_test>max_test:
            within_size=max_test
        elif is_pair is False:
            within_size=n_test
        else:
            within_size = int(2**(len(format(n_test,'b'))-1))
        excess=max(n_test-within_size, 0)

        subsets = dict()
        subsets_comb = dict()
        t_indices = []
        c_indices = []
        if is_pair is False:
            new = list(a['index'])
            if random is True:
                shuffle(new)
            t=np.array(new[:within_size])
            c=np.array(new[-within_size:])
        elif (max_test>=2 and n_test>=2 and is_pair is True)&(min(len(segments['t0']),len(segments['t1']))>0):  #Last condition needed to handle dedicated segments
            for c in ['t0','t1']:
                comb = np.array(list(combinations(segments[c], int(within_size))))
                if random is True:
                    shuffle(comb)
                subsets[c] = comb[0]
                subsets_comb[c] = np.array(list(combinations(subsets[c], int(within_size/2))))
                if random is True:
                    shuffle(subsets_comb[c])

            for i in ['t0','t1']:
                t_indices.append(subsets_comb[i][0])
                c_indices.append(subsets[i][~np.isin(subsets[i],subsets_comb[i][0])])

            t=np.concatenate(t_indices)
            c=np.concatenate(c_indices)
        else:
            t=np.array([])
            c=np.array([])
        assigned_indices = dict(t=t, c=c, excess=excess)
    else:
        new = list(a['index'])
        size = len(new)
        max_test = math.floor(size/(variants+1))
        within_size = min(max_test,n_test)
        if random is True:
            shuffle(new)
        assigned_indices=dict()
        assigned_indices['c']=np.array(new[:within_size])
        treatments = dict()
        assigned_indices['excess'] = max(0, n_test-within_size*(variants+1))
        for i in range(variants):
            assigned_indices['t{}'.format(i+1)] = np.array(new[within_size*i:within_size*(i+1)])
    return assigned_indices

def assign_segments(tests, n_test=4, random=True, segment_size=32, empty_first=True, variants=1):
    units, pairs = get_test_units(tests)
    units_empty = units.get(99) #99 is a dummy variable to represent empty segments
    ukey = list(units.keys())
    if units_empty is not None:
        ukey.remove(99)
        if empty_first is True:
            ukey = [99]+ukey
        else:
            ukey = ukey + [99]
    control = np.array([])
    treatment = np.array([])

    new = np.zeros(segment_size).astype(int)
    if variants==1:
        for i in ukey:
            if n_test==0:
                break
            if i==99:
                is_pair=False
            else:
                is_pair=True
            u = assign_test_unit_index(units[i], n_test=n_test, random=random, is_pair=is_pair)
            control = np.append(control, u['c'])
            treatment = np.append(treatment, u['t'])
            n_test = max(u['excess'],0)
        new[control.astype(int)]=-1
        new[treatment.astype(int)]=1
    else:
        u = assign_test_unit_index(units[99], n_test=n_test, random=random, is_pair=False, variants=variants)
        new[np.append(control, u['c']).astype(int)]=-1
        for i in range(variants):
            new[np.append(treatment, u['t{}'.format(i+1)]).astype(int)]=i+1
    return new

def load_experiment_table(exp_table='google_sheets.ab_experiment_table',
                          unique_fields=['name','actor_type','iteration'],
                          platforms=['android','iphone','ipad','web'],
                          date_regex='(time|date)',
                          start_dates = ['start_time','estimated_start_time'], #order by priority
                          end_dates = ['end_time','estimated_end_time'] #order by priority
                            ):
    experiments = redshift('select * from {0} limit 10000'.format(exp_table)) #Just putting at cap just in case of mistake
    date_col = experiments.filter(regex='(time|date)').columns
    experiments[date_col] = experiments[date_col].apply(pd.to_datetime)
    experiments[start_dates[0]] = experiments[start_dates[0]].combine_first(experiments[start_dates[1]])
    experiments[end_dates[0]] = experiments[end_dates[0]].combine_first(experiments[end_dates[1]])
    for p in platforms:
        experiments[p] = experiments['platform'].str.contains(p).astype(int)
    #Add Checks
    #duplication checks

    for p in platforms:
        cnt = experiments.groupby(unique_fields+[p]).size()
        dupes = cnt[cnt>1]
        if len(dupes)>0:
            print("WARNING: Duplicate experiments found")
            print(dupes)

    return experiments

def get_experiment_dates(post_start_date,
        post_end_date=None, #if available
        duration_days=14, #use if end_date not provided
        #platform=['android','iphone','ipad','web'],
        #status=['Active', 'Rolled Back', 'Graduated', 'Segments Reserved'],
        pre_days=14,
        pre_start_date=None, #Custom pre start date, default 14 days before start_day
        pre_end_date=None, #custom pre end date, else pre_start_date+14-1
        buffer=7,
        ):

    #Assign time of experiment
    nat = np.datetime64('NaT')
    post_start_date = pd.to_datetime(post_start_date).date()
    if pd.isnull(post_end_date):
        post_end_date = post_start_date + pd.to_timedelta(duration_days-1, unit='d')
    else:
        post_end_date = pd.to_datetime(post_end_date).date()

    post_end_date = post_end_date + pd.to_timedelta(buffer, unit='d')

    if pd.isnull(pre_start_date):
        pre_start_date = post_start_date-pd.to_timedelta(pre_days, unit='d')
    else:
        pre_start_date = pd.to_datetime(pre_start_date).date()

    if pd.isnull(pre_end_date):
        pre_end_date = pre_start_date+pd.to_timedelta(pre_days-1, unit='d')
    else:
        pre_start_date = pd.to_datetime(pre_end_date).date()

    return dict(pre_start_date=pre_start_date,
                pre_end_date=pre_end_date,
                post_start_date=post_start_date,
                post_end_date=post_end_date,
               )

def overlap_dates(dt1:dict, dt2:dict):
    range1 =pd.date_range(dt1['pre_start_date'],dt1['pre_end_date']).append(pd.date_range(dt1['post_start_date'],dt1['post_end_date']))
    range2 =pd.date_range(dt2['pre_start_date'],dt2['pre_end_date']).append(pd.date_range(dt2['post_start_date'],dt2['post_end_date']))
    return len(range1.intersection(range2))

def partition_experiments(experiments,
                          actor_type='user',
                          assigned=['Active', 'Rolled Back', 'Graduated', 'Segments Reserved'],
                          upcoming=['Initial Assignment', 'Needs Segments'],
                          sort_by=['priority', 'start_time'],
                          sort_asc=['asc','asc']
                         ):
    assigned = experiments[experiments.status.isin(assigned)&(experiments.actor_type==actor_type)]
    upcoming = experiments[experiments.status.isin(upcoming) & (experiments.actor_type==actor_type) &
                           (experiments.segments_per_variant>0) #Must have this entered
                          ].sort_values(by=sort_by, ascending=sort_asc)

    return assigned, upcoming

def get_test_array(experiments,
        segments,
        label2val={'V1':1,'-':0,'DC':-1, 'V2':2},
        ):
    return pd.DataFrame(np.array(experiments[segments].replace(label2val).fillna(0).astype(int)), columns=range(len(segments)), index=experiments.index)

def populate_upcoming_segments(experiments,
                               actor_type='user',
                               pre_days=14,
                               buffer=7,
                               num_segments=32,
                               test_id=['name','actor_type','platform','iteration','start_time'],
                               label2val={'V1':1,'-':0,'DC':-1, 'V2':2},
                               random=False, #random selection of which segments go to control or treatment
                               alternate_order=True,#Needed because otherwise the early segments will always get treatment if we don't randomize.
                              ):
    experiments['test_dates'] = experiments.apply(lambda x: get_experiment_dates(post_start_date=x['start_time'], post_end_date=x['end_time'], duration_days=x['duration_days'], pre_days=pre_days, buffer=buffer), axis=1)
    assigned, upcoming = partition_experiments(experiments, actor_type=actor_type)

    segments = ['t{}'.format(i) for i in range(1,num_segments+1)] #this is very adhoc, maybe parametize but not needed in near future.

    for i, u in upcoming.iterrows():
        assigned['overlap_days'] = assigned['test_dates'].apply(lambda x: overlap_dates(x, u['test_dates']))
        eligible_assigned = assigned[assigned.overlap_days>0]
        eligible_assigned_arr = get_test_array(eligible_assigned, segments, label2val=label2val)#pd.DataFrame(np.array(eligible_assigned[segments].replace({'V1':1,'-':0,'DC':-1}).fillna(0).astype(int)), columns=range(num_segments), index=eligible_assigned.index) #I use an np.array to handle this so kinda funky
        new_segments = assign_segments(eligible_assigned_arr, n_test=u['segments_per_variant'], empty_first=True, segment_size=num_segments, random=random, variants=u['test_variants'])
        if alternate_order is True:
            multiplier = list([-1,1])[i%2]
            new_segments = new_segments*multiplier
        assigned = assigned.append(u)
        assigned.loc[i,segments] = new_segments
        if np.abs(new_segments).sum()==0:
            print("WARNING: No segment assigned:"+u['name'])
    val2label = {v: k for k, v in label2val.items()}
    assigned[segments] = assigned[segments].replace(val2label)

    return assigned.loc[upcoming.index][test_id+segments].sort_index()

def plot_experiment_schedule(experiments, start, stop, height=None, width=None):
    df=[]
    r = pd.date_range(start, stop)
    for i, e in experiments.iterrows():
        d = get_experiment_dates(post_start_date=e['start_time'], post_end_date=e['end_time'], duration_days=e['duration_days'], pre_days=0, buffer=0)
        if len(r.intersection(pd.date_range(d['post_start_date'],d['post_end_date'])))>0:
            df.append(dict(Task=e['name'], Start=d['post_start_date'], Finish=d['post_end_date'], Experiment=e['status']))

    colors = {'Active':'#2ca02c',  #cooked asparagus green
              'Rolled Back':'#d62728',  #brick red,
              'Graduated':'#7f7f7f',  #middle gray
              'Segments Reserved':'#17becf',   #blue-teal
              'Initial Assignment':'#072859',  #custom blue
              'Needs Segments':'#ff7f0e',  #safety orange
             }
    fig = ff.create_gantt(df, colors=colors, index_col='Experiment', title='Experiments',
                          show_colorbar=True, bar_width=0.2, showgrid_x=True, showgrid_y=True, height=height, width=width)
    fig.layout.xaxis.dtick = 86400000.0*3
    return fig
                          
