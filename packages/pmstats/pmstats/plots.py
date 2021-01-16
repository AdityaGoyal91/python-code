import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime, timedelta
from .abtest import *
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import plot, init_notebook_mode, iplot

idx = pd.IndexSlice
colors = [ '#1f77b4',  #muted blue, https://stackoverflow.com/questions/40673490/how-to-get-plotly-js-default-colors-list
            '#ff7f0e',  #safety orange
            '#2ca02c',  #cooked asparagus green
            '#d62728',  #brick red
            '#9467bd',  #muted purple
            '#8c564b',  #chestnut brown
            '#e377c2',  #raspberry yogurt pink
            '#7f7f7f',  #middle gray
            '#bcbd22',  #curry yellow-green
            '#17becf',   #blue-teal
            '#072859'  #custom blue
            ]

def plot_prepost(pre, post, metric, datefield = 'event_date', figsize=[20,6], title=None):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title or metric, size=16, fontweight='bold')
    gs = GridSpec(2, 3)
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[0,1])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[1,1])
    ax4 = plt.subplot(gs[:,2])

    pre_stats = get_relative_diff(pre,'pre', metric, experiment_unit=datefield)['data'].reset_index()
    post_stats = get_relative_diff(post,'post', metric, experiment_unit=datefield)['data'].reset_index()

    sns.swarmplot(x='name',y='rel_diff', data=pd.concat([pre_stats, post_stats]),ax=ax4, hue='name', palette="Set2", linewidth=1,edgecolor='gray')
    sns.boxplot(x='name',y='rel_diff', data=pd.concat([pre_stats, post_stats]),ax=ax4, width=0.1, hue='name', palette="Set2", boxprops=dict(alpha=.3))
    handles, labels = ax4.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2])
    sns.lineplot(x=datefield, y=metric, hue="test_group", data=pre, markers=True, ax=ax0)
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True,  borderaxespad=0.)
    sns.lineplot(x=datefield, y=metric, hue="test_group", data=post, markers=True, ax=ax1)
    ax1.legend().remove()
    ylim = ax0.get_ylim()+ax1.get_ylim()
    ax0.set_ylim([min(ylim), max(ylim)])
    ax1.set_ylim([min(ylim), max(ylim)])
    ax0.set_ylabel('value')
    ax1.set_ylabel('value')

    sns.lineplot(x=datefield, y='rel_diff', data=pre_stats, ax=ax2, color='black')
    sns.lineplot(x=datefield, y='rel_diff', data=post_stats, ax=ax3, color='black')
    ylim = ax2.get_ylim()+ax3.get_ylim()
    ax2.set_ylim([min(ylim), max(ylim)])
    ax3.set_ylim([min(ylim), max(ylim)])
    ax2.set_title('PRE')
    ax3.set_title('POST')
    ax4.set_title('% DIFFERENCE BY PRE AND POST DAYS')
    for ax in [ax2, ax3, ax4]:
         ax.yaxis.set_major_formatter(FuncFormatter('{0:.2%}'.format))

    fig.autofmt_xdate(rotation=90)
    return fig

def plotly_prepost(pre, post, metric, datefield = 'event_date', height=400, width=1000, title=None, marker_mode='lines+markers', ab_result=None):
    color_map = dict(control=colors[0],
                     test=colors[1],
                     pre='rgb(7,40,89)',   #'rgb(7,40,89)',
                     post=colors[5],
                     )
    boxname = ['Pre','Post']
    pre_stats = get_relative_diff(pre,'pre', metric, experiment_unit=datefield)['data'].reset_index()
    post_stats = get_relative_diff(post,'post', metric, experiment_unit=datefield)['data'].reset_index()
    fig = tools.make_subplots(rows=2, cols=3, specs=[[{}, {}, {'rowspan': 2}], [{}, {}, None],],
                                print_grid=False,
                                shared_yaxes=True,
                                subplot_titles=['Pre', 'Post', None, None, None]
                                )

    for i, df in enumerate((pre, post)):
        stats = get_relative_diff(df,'stats', metric, experiment_unit=datefield)['data'].reset_index()
        for group in ['CONTROL', 'TEST']:
            showlegend = (i==0)
            fig.append_trace(go.Scatter(
                                   x=df[df.test_group==group][datefield],
                                   y=df[df.test_group==group][metric],
                                   mode=marker_mode,
                                   line=dict(color=color_map[group.lower()]),
                                   marker=dict(size=6),
                                   name=group,
                                   legendgroup=group,
                                   showlegend=showlegend,
                                   yaxis='y1',
                                   ), 1, i+1)

        fig.append_trace(go.Scatter(
                                   x=stats[datefield],
                                   y=stats['rel_diff'],
                                   mode=marker_mode,
                                   line=dict(color=color_map[boxname[i].lower()]),
                                   marker=dict(size=6),
                                   name='Delta (%)',
                                   legendgroup=group,
                                   showlegend=False,
                                   yaxis='y4',

                                   ), 2, i+1)
        fig.append_trace(go.Box(
                            text=stats[datefield],
                            y=stats['rel_diff'],
                            name = boxname[i],
                            jitter = 0.3,
                            pointpos = 0,
                            boxpoints = 'all',
                            marker = dict(color = color_map[boxname[i].lower()]),
                            yaxis='y3',
                            ), 1, 3)
    fig['layout'].update(height=height,
                            width=width,
                            title=(title or metric),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis4=dict(title='Pre'),
                            xaxis5=dict(title='Post'),
                            yaxis1=dict(title='Value'),
                            yaxis3=dict(title='Delta %', tickformat='.1%', zeroline=True, zerolinecolor='gray'),
                            yaxis4=dict(title='Delta %', tickformat='.1%', zeroline=True, zerolinecolor='gray'),
                            yaxis5=dict(tickformat='.1%', zeroline=True, zerolinecolor='gray')
                            )
    if ab_result is not None:
        l=list(ab_result[ab_result['Metric']==metric][['PrePost Delta (%)', 'p-value', 'Net Impact']].iloc[0])
        fig['layout'].update(xaxis3=dict(title=dict(text='PrePost Delta (%): {0:.1%} p-value: {1:.3f}\nNet Impact: {2:.2f}'.format(*l),font=dict(size=10))))

    #fig.layout['yaxis3'].update(tickformat='.1%')
    #fig.layout['yaxis2'].update(tickformat='.1%')

    fig.layout.yaxis3.update(anchor='x3', side='right')
    fig.layout.legend.update(x=1.1)
    fig.data[2].update(xaxis='x1')
    fig.data[6].update(xaxis='x2')

    fig.data[3].update(yaxis='y3')
    fig.data[7].update(yaxis='y3')
    return fig

def plotly_metrics_delta(result,
                            title='AB Result',
                            metric_col='Metric',
                            mid_col='PrePost Delta (%)',
                            lo_col='PrePost Delta LCL (%)',
                            hi_col='PrePost Delta UCL (%)',
                            height=None,
                            margin=dict(l=400),
                            xlim=[-1,1],
                            tickformat='.1%',
                            ):
    data=[]
    for i, row in result[::-1].iterrows():
        metric = row[metric_col]
        mid = row[mid_col]
        lo = row[lo_col]
        hi = row[hi_col]
        if lo>0:
            color='#2ca02c'
        elif hi <0:
            color='#d62728'
        else:
            color='#7f7f7f'
        data.append(go.Box(x=[lo,lo,mid,hi,hi],
                            name=metric,
                            marker=dict(color=color)))
    layout = go.Layout(
        title= title,
        xaxis= dict(
            title=mid_col,
            range=xlim,
            nticks=10,
            tickformat=tickformat,
            gridwidth=2,
            zeroline=True,
            zerolinecolor='Black',
        ),
        yaxis=dict(
            gridwidth= 2,
            dtick=1,
            #gridcolor='blue',
        ),
        showlegend= False,
        margin = margin,
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)',
        )
    fig= go.Figure(data=data, layout=layout)
    if height is not None:
        fig['layout'].update(height=height)
    return fig

def plotly_lineplot(x, y, data, hue,
                    title='',
                    mode='lines+markers',
                    marker=dict(size=6),
                    color_map=None,
                    xaxis=None,
                    yaxis=None,
                    height=400,
                    width=800):
    if hue is None:
        hues=[y]
    else:
        hues = sorted(list(set(data[hue])))
    if color_map is None:
        color_map = dict(zip(hues,colors[:len(hues)]))
    trace = []
    for h in hues:
        if hue is None:
            df = data
        else:
            df = data[data[hue]==h]
        trace.append(go.Scatter(x=df[x],
                    y=df[y],
                    mode=mode,
                    line=dict(color=color_map[h]),
                    marker=marker,
                    name=h,
                    ))

    if xaxis is None:
        xaxis = dict(title=x)
    if yaxis is None:
        yaxis = dict(title=y)
    layout = go.Layout(
        title=title,
        height=height,
        width=width,
        xaxis=xaxis,
        yaxis=yaxis,
        )
    fig = go.Figure(data=trace, layout=layout)
    return fig


def plot_yoy_trace(date,
                  column,
                  df,
                  name,
                  xaxis='x1',
                  yaxis='y1',
                  rolling=1,
                  window=365,
                  yoy_range=None,
                  percent=True,
                  show_plot=True,
                  height=None,
                  width=None,
                 ):

    fig = tools.make_subplots(rows=1, cols=2, shared_xaxes=True,
                                print_grid=False,
                                subplot_titles=[name, name + ' (YOY%)']
                                )

    x = df[date]
    y1 = df[column].rolling(rolling).mean()
    y0 = y1.shift(window)
    delta = (y1-y0)
    if percent is True:
        delta = (y1-y0).div(y0)

    #delta_std = delta.rolling(rolling).mean()/np.sqrt(rolling)
    c = delta.apply(lambda y: 'red' if y<0 else 'blue')
    bar_trace = go.Scatter(
        x=x[window:],
        y=delta[window:],
        marker=dict(color=c),
        name=name,
        xaxis=xaxis,
        yaxis=yaxis,
        showlegend=False,
        )

    line_trace_curr = go.Scatter(
        x = x[window:],
        y = y1[window:],
        name= 'Current',
        line=dict(color='black', width=2),
        xaxis=xaxis,
        yaxis=yaxis,
        showlegend=False,
            )

    line_trace_prev = go.Scatter(
        x = x[window:],
        y = y0[window:],
        name= 'Previous',
        line=dict(color='black', width=2, dash='dot'),
        xaxis=xaxis,
        yaxis=yaxis,
        showlegend=False,
            )
    traces = [bar_trace, line_trace_curr, line_trace_prev]
    fig.append_trace(line_trace_curr, row=1, col=1)
    fig.append_trace(line_trace_prev, row=1, col=1)
    fig.append_trace(bar_trace, row=1, col=2)

    if percent is True:
        tickformat = '.1%'
        ytitle = "YOY %"
    else:
        tickformat = None
        ytitle = "YOY Diff"

    fig['layout'].update(title=name,
                            height=height,
                            width=width,
                            xaxis=dict(
                                title=date,
                                nticks=10,
                                ),
                            xaxis2=dict(
                                title=date,
                                nticks=10,
                                ),
                            yaxis2=dict(
                                range=yoy_range,
                                title=ytitle,
                                tickformat=tickformat,
                                ),
                        )



    if show_plot is True:
        iplot(fig)
    return fig#, traces

def plotly_delta_ts(time,
        y,
        df,
        hue,
        title=None,
        height=None,
        width=None,
        hourly_interval=None,
        orientation='h',
        annotations=None,
        box_intervals=None):
    fig = dict()
    df_tmp = df.copy()
    if hourly_interval is not None:
        df_tmp[time] = df_tmp[time].apply(lambda x: pd.to_datetime(x.date())+timedelta(hours=math.floor(x.hour/hourly_interval)*hourly_interval))

    df_tmp = df_tmp.groupby([hue, time])[y].sum().reset_index()
    fig[1] = plotly_lineplot(x=time,y=y,data=df_tmp, hue=hue, mode='lines')
    reldiff = get_relative_diff(df_tmp,'stats', y, experiment_unit=time)
    df_diff = reldiff['data'].reset_index()
    fig[2] = plotly_lineplot(x=time,y='rel_diff',data=df_diff,hue=None, mode='lines')
    # fig[2]['data'][0]['line'].update(dash='dot')
    fig[2]['data'][0].update(name='Relative Diff (%)')
    fig[2]['data'][0]['line'].update(color='gray')
    #fig2['data'][0].update(mode='lines')
    box_space = 0 if box_intervals is None else 1


    if orientation=='h':
        rows=1
        cols=2 + box_space
        chart2row, chart3row = 1, 1
        chart2col, chart3col = 2, 3
    else:
        rows=2 + box_space
        cols=1
        chart2row, chart3row = 2, 3
        chart2col, chart3col = 1, 1

    fig['all'] = tools.make_subplots(rows=rows, cols=cols, shared_xaxes=True,
                            print_grid=False,
                            #subplot_titles=[name, name + ' (YOY%)']
                                )
    fig['all'].append_trace(fig[1]['data'][0], row=1, col=1)
    fig['all'].append_trace(fig[1]['data'][1], row=1, col=1)
    fig['all'].append_trace(fig[2]['data'][0], row=chart2row, col=chart2col)

    fig['all']['layout'].update(title=title or y, height=height, width=width)
    fig['all']['layout']['yaxis2'].update(tickformat='.1%', title='Delta (%)')
    fig['all']['layout']['yaxis1'].update(title='Value')

    y2min = np.min(fig['all']['data'][2]['y'])
    y2max = np.max(fig['all']['data'][2]['y'])

    notes = list()
    if annotations is not None:
        for dt, note in annotations.items():
            dt = pd.to_datetime(dt)
            notes.append(dict(x=dt, y=1.05*y2max, xref='x2', yref='y2',text=note, showarrow=False, font=dict(size=10)))
            fig['all'].append_trace(go.Scatter(
                        x = [dt, dt],
                        y = [y2min, y2max],
                        mode='lines',
                        line=dict(color='black', width=1, dash='dot'),
                        xaxis='x1',
                        yaxis='y1',
                        text = 'Test',
                        #textposition='top',
                        showlegend = False,
                ), row=chart2row, col=chart2col)
        fig['all']['layout']['annotations'] = notes

    if box_intervals is not None:
        shapes = []
        for name, i in box_intervals.items():
            #print(reldiff.columns)
            df_i =df_diff[df_diff[time].between(*i)]
            fig['all'].append_trace(go.Box(
                                    text=df_i[time],
                                    y=df_i['rel_diff'],
                                    name = name,
                                    jitter = 0.3,
                                    pointpos = 0,
                                    boxpoints = 'all',
                                    marker = dict(color ='gray'),
                                    showlegend=False,
                                    #yaxis='y3',
                                    ), row=chart3row, col=chart3col)
            shapes.append(dict(
                type = 'rect',
                xref = 'x2',
                yref = 'y2',
                x0 = i[0],
                y0 = y2min,
                x1 = i[1],
                y1 = y2max,
                fillcolor = '#d3d3d3',
                opacity=0.3,
                line=dict(width=1, dash='dot'),
                layer='below',
                ))

            notes.append(dict(x=pd.to_datetime(pd.Series(i)).mean(), y=1.05*y2max, xref='x2', yref='y2',text=name, showarrow=False, font=dict(size=10)))


        fig['all']['layout']['yaxis3'].update(tickformat='.1%', title='Delta (%)')
        fig['all']['layout']['shapes'] = shapes
        fig['all']['layout']['annotations'] = notes


    return fig['all']

def get_summary_table(columns=['Parameter','Values'], **kwargs):
    details = pd.DataFrame(list(kwargs.items()), columns=columns)
    details[columns[0]]= details[columns[0]].str.replace('_',' ').str.title()

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(details.columns),
                    fill= dict(color='#A43820'),
                    align='center',
                    font = dict(size = 18, color = "#ffffff" ),
                    height = 35),
        cells=dict(values=[details.Parameter, details.Values],
                   fill=dict(color='#f4e1d2'),
                   align='center',
                   height = 30))
    ])
    return fig, kwargs
