from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pprint
import pandas as pd
import numpy as np
from pmstats import assignment
from datetime import datetime


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SPREADSHEET_ID = '1Jcs5jcDyR2z2B-u4950sbvx5U46fRhjrF6DKmlzgORM'
RANGE_NAMES = ['Form Responses 1!A:H', 'ab_experiment_table!A:AV']


# takes a pandas df and converts all columns to strings/object types
def pandas_cast_columns(df, cols=None, col_type='str', _all=True):

    if not _all and not cols:
        # todo raise error.
        pass
    elif _all:
        cols = df.columns

    for c in cols:
        if col_type in ('date', 'timestamp', 'datetime'):
            df[c] = pd.to_datetime(df[c])
        else:
            df[c] = df[c].astype(col_type)


def googlesheet_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    try:
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
    except:
        pass

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '.credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    return service


def get_googlesheet_values(service, sheet_id=None, ranges=None):

    if sheet_id == None or ranges == None:
        #todo raise error
        return

    # Call the Sheets API
    sheet = service.spreadsheets()

    for r in ranges:
        result = sheet.values().get(spreadsheetId=sheet_id,
                                    range=r).execute()
        values = result.get('values', [])
    return [sheet.values()
                 .get(spreadsheetId=sheet_id, range=r)
                 .execute()
                 .get('values', [])
            for r in ranges]


def post_googlesheet_values(service, sheet_id=None, _range=None, df=None):
    if sheet_id == None or _range == None:
        #todo raise error
        return

    # remove datatypes -- everything should be an object
    pandas_cast_columns(df)

    service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        valueInputOption='RAW',
        range=_range,
        body=dict(
            majorDimension='ROWS',
            values=df.values.tolist())
    ).execute()
    print('Sheet successfully Updated')




def run(attempt=0, debug=False, random=True, buffer=7):
    try:
        print(f'attempt # {attempt+1}')
        print('setting up service')
        service = googlesheet_service()
        print('fetching data')
        data = get_googlesheet_values(service, sheet_id=SPREADSHEET_ID, ranges=RANGE_NAMES)
        form_df = pd.DataFrame(data[0])
        form_df.columns = form_df.iloc[0]
        form_df.columns = [c.lower() for c in form_df.columns]
        form_df.drop(form_df.index[0], inplace=True)
        form_df.timestamp = pd.to_datetime(form_df.timestamp)
        form_df['expid'] = form_df.timestamp.astype('int64')
        form_df.set_index('expid', inplace=True, drop=True)
        form_df.sort_index(inplace=True)

        experiement_df = pd.DataFrame(data[1])
        experiement_df.columns = experiement_df.iloc[0]
        experiement_df.columns = [c.lower().strip().replace(' ', '_')
                                for c in experiement_df.columns]
        experiement_df.drop(experiement_df.index[0], inplace=True)
        try:
            experiement_df.expid = experiement_df.expid.astype('int64')
        except:
            experiement_df.expid = pd.to_datetime(
                experiement_df.expid).astype('int64')
        experiement_df.set_index('expid', inplace=True, drop=True)
        experiement_df.sort_index(inplace=True)

        print('building dataframes')

        date_cols = ['estimated_start_time', 'start_time',
                    'estimated_end_time', 'end_time']
        int_cols = ['iteration', 'duration_days', 'priority',
                    'test_variants', 'segments_per_variant']

        for c in date_cols:
            experiement_df[c] = pd.to_datetime(experiement_df[c], errors='coerce')
        for c in int_cols:
            experiement_df[c] = experiement_df[c].astype('int32')

        max_expid = experiement_df.index[-1]

        new_form = form_df[form_df.index > max_expid].drop(columns=['timestamp'])
        new_form.sort_index(inplace=True)
        new_form.columns = experiement_df.columns[:7]

        default_dict = {'priority': 1,
                        'test_variants': 1,
                        'segments_per_variant': 2,
                        'duration_days': 17,
                        'status': 'Needs Segments'}

        segment_columns = ['t'+str(s) for s in range(1, 33)]  # todo make variable

        for s in segment_columns:
            default_dict[s] = '-'
        for c in experiement_df.columns:
            if c not in new_form.columns:
                new_form[c] = default_dict.get(c)

        updated_df = experiement_df.append(new_form, sort=False)
        updated_df.sort_index(inplace=True)
        print('loading experiements')
        experiments = assignment.load_experiment_table(
            unique_fields=['expid'],
            experiments=updated_df.copy().reset_index())

        test_cols = ['expid'] + \
            [c for c in updated_df.columns if c not in segment_columns]
        tmp = experiments.copy().reset_index()
        """
        return {'updated assignments': None,
                'new assignments': tmp,
                       'experiments': experiments,
                       'updated experiment table': updated_df,
                       'new form responses': new_form,
                       'ab experiment table': experiement_df,
                       'form responses': form_df
                       }
                       """
        print('updating assignments')
        new_assignments = assignment.populate_upcoming_segments(
            tmp, buffer=buffer, random=random, test_id=test_cols).copy()
        
        new_assignments.set_index('expid', inplace=True)
        
        
        #if data - pd.to_datetime("now")
        #new_assignments.status = 'Initial Assignment'


        
        updated_assignments = updated_df.copy().drop(new_assignments.index)\
                                        .append(new_assignments, sort=False)
        updated_assignments.sort_index(inplace=True)
        print('updating sheet')

        #todo: Add check
        #if data is bad:
        #    raise error # retry 

        post_googlesheet_values(service, sheet_id=SPREADSHEET_ID,
                                _range='ab_experiment_table!A2:ZZ', df=updated_assignments.reset_index())

        # return all dataframes for easy qa/debugging
        if debug:
            return {'updated assignments': updated_assignments,
                'new assignments': new_assignments,
                'experiments': experiments,
                'updated experiment table': updated_df,
                'new form responses': new_form,
                'ab experiment table': experiement_df,
                'form responses': form_df
                }
    except Exception as e:
        print(e)
        if attempt < 3:
            run(attempt=attempt+1, debug=debug, random=random, buffer=buffer)
        else:
            raise e
