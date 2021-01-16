'''
Function to split dw_campaign_calendar into dataframes that the Prophet model can ingest.
Inputs:
df: dataframe with result from dw_campaign_calendar. This df will need to be filtered by the
    desired campaign before being input.
    Example for MDD days: campaign_calendar.loc[campaign_calendar['campaign_type']=='MDD']
min_date, max_date: minimum and maximum dates in the returned dataframe
Output: Dataframe with two columns
        1) ds: column of sequential dates, one record for each date.
               Sequence will be from min_date to max_date inclusive
        2) campaign: Binary column with 1=campaign occured on date and 0=campaign did not occur on date
'''

import warnings
warnings.filterwarnings('ignore',)

import pandas as pd
import numpy as np
import math

from datetime import datetime
from datetime import timedelta


def split_campaign_calendar(df, min_date, max_date):
    full_df=pd.DataFrame()
    full_df['ds']=pd.date_range(start=min_date, end=max_date)
    full_df['campaign']=0

    for r in range(0, df.shape[0]):
        campaign_type = df.iloc[r]['campaign_type']
        start=df.iloc[r]['campaign_started_at']
        end=df.iloc[r]['campaign_ended_at']
        day_diff = (end-start).days
        if day_diff>0:
            for i in range(1,day_diff+1):
                day_add = start + timedelta(days=i)
                new_row = pd.DataFrame({'campaign_type': [campaign_type],
                                        'campaign_started_at': [day_add],
                                        'campaign_ended_at': [day_add + timedelta(hours=23, minutes=59, seconds=59)]
                                       })
                df = pd.concat([df, new_row], axis=0, ignore_index=True)

    df = df.drop(columns=['campaign_ended_at'])
    df = df.rename(index=str, columns={"campaign_started_at": "campaign_date"})
    df = df.loc[df['campaign_date']>=pd.to_datetime("2015-01-01")] # specific for prophet forecasts. Can be removed.
    df.reset_index(inplace=True, drop=True)
    full_df.loc[full_df['ds'].isin(df['campaign_date']), 'campaign']=1
    return full_df
