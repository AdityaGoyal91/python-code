# Import Libraries needed for API, and Pandas
import requests
import pandas as pd
import numpy as np
import configparser
import keyring
from pmutils.creds import read_config
import json

# Example Code
#   looker = lookerAPIClient()
#   looker.authorize()
#   look = looker.runlook(look_id=6536)
#   print(look['sql'])


class lookerAPIClient:
    """
    Very Basic Looker API class allowing us to access the data from a given Look ID
    #Read config file with Looker API and Database connection information
    config = configparser.RawConfigParser(allow_no_value=True)
    config.read('/../../.poshmark.cfg')
    #Initialize the Looker API Class with the data in our config file (which is stored in a neighboring file 'config')
    x = lookerAPIClient(
            api_host      = config.get('looker', 'api_host'),
            api_client_id = config.get('looker', 'api_client_id'),
            api_secret    = config.get('looker', 'api_secret'),
            api_port      = config.get('looker', 'api_port')
            )
    #Use the API to get our training/'test' dataset and our new 'validation' dataset we will predict upon
    historicalCustomers = x.runLook('5373',limit=10000)
    newCustomers = x.runLook('5373',limit=10000)
    historicalCustomersDF = pd.DataFrame(historicalCustomers)
    newCustomersDF  = pd.DataFrame(newCustomers)
    """
    # TODO : Create authorization as sub class method
    # : Initiate variables from config inside class constructor
    # : Build relevant methods out

    def __init__(self, initialize_type = 'config', config=read_config(), section='looker_api', uri_stub='/api/3.1/'):

        if initialize_type == 'keyring':
            self.api_host = 'https://poshmark.looker.com'
            api_client_id = keyring.get_password('looker', 'client_id')
            api_secret    = keyring.get_password('looker', 'secret')
            self.api_port = '19999'
            self.api_login = 'https://poshmark.looker.com:19999/login'
        else :
            self.api_host = config.get(section, 'looker_host')
            self.api_port = config.get(section, 'api_port')
            self.api_login = config.get(section, 'looker_login')
            api_client_id = config.get(section, 'looker_client_id')
            api_secret = config.get(section, 'looker_secret')

        self.auth_request_payload = {
            'client_id': api_client_id, 'client_secret': api_secret}

        self.uri_stub = uri_stub
        self.uri_full = ''.join([self.api_host, ':', self.api_port, self.uri_stub])

    def authorize(self):
        try:
            response = requests.post(
                self.api_login, params=self.auth_request_payload)
            print("Authorization response: ", response.status_code)
            authData = response.json()
            self.access_token = authData['access_token']
            self.auth_headers = {
                'Authorization': 'token ' + self.access_token,
            }
            # print(self.auth_headers)
        except:
            print("Error occured while authorizing")

    def getlook(self, look_id=''):
        response = requests.get(
            self.uri_full + 'looks/' + str(look_id), headers=self.auth_headers)
        return response.json()

    def getquery(self, query_id):
        response = requests.get(
            self.uri_full + 'queries/' + str(query_id), headers=self.auth_headers)
        return response.json()

    def getsql_runner_query(self, slug=''):
        # print(self.uri_full + 'sql_queries/' + str(slug))
        response = requests.get(
            self.uri_full + 'sql_queries/' + str(slug), headers=self.auth_headers)
        return response.json()

    def runlook(self, look_id='', result_format='json_detail'):
        response = requests.get(self.uri_full + 'looks/' + str(look_id) +
                                '/run/' + result_format, headers=self.auth_headers)
        return response.json()

    # testing some code
    def get(self, call=''):
        response = requests.get(self.uri_full + call,
                                headers=self.auth_headers)
        return response.json()

    def runLookPD(self, look, limit=100):
        optional_arguments = '?' + 'limit=' + str(limit)
        return pd.DataFrame(self.get('/'.join(['looks', str(look), 'run', 'json'])+optional_arguments))

    def getLookSQL(self, query_id, limit='100'):
        return requests.get(self.uri_full + 'queries/{query_id}/run/{result_format}?limit={limit}'
                       .format(query_id=query_id,result_format = 'sql',limit=limit),headers=self.auth_headers).text
