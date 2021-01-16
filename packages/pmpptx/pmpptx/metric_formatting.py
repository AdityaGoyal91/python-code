import datetime
import pandas as pd

def act_kpi_format(df_to_update):

    copy = df_to_update.copy()

    for c in range(0, df_to_update.shape[1]):
        if copy.columns[c] == 'acq_month':
            for r in range(0, df_to_update.shape[0]):
                upd_str = df_to_update.iloc[r, c]
                copy.iloc[r, c] = upd_str

        ## Sums and Counts first ##

        if copy.columns[c] == 'spend_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c]/1000000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'imps_1k_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c]/1000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'clicks_1k_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.2f}'.format(df_to_update.iloc[r, c]/1000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'installs_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.2f}'.format(df_to_update.iloc[r, c]/1000000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'users_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'gmv_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c]/1000000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'buyers_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'listers_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'sellers_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str

        ## Cost per metrics ##

        if copy.columns[c] == 'cpm_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpc_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpi_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpu_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpnb_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpnl_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpns_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str

        ## Conversion Metrics ##

        if copy.columns[c] == 'ctr_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'ipc_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'upi_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'upc_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'buyer_act_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'lister_conv_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'seller_conv_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'user_ret_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'gmv_ret_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'buyer_act_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'lister_act_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'seller_act_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'me_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str

    return copy

def yoy_kpi_format(df_to_update):

    copy = df_to_update.copy()

    for c in range(0, df_to_update.shape[1]):
        if copy.columns[c] == 'acq_month':
            for r in range(0, df_to_update.shape[0]):
                upd_str = df_to_update.iloc[r, c]
                copy.iloc[r, c] = upd_str
        else:
            for r in range(0, df_to_update.shape[0]):
                upd_str = '\n(' + '{0:.0%}'.format(df_to_update.iloc[r, c]) + ' YoY)'
                df_to_update.iloc[r, c]
                copy.iloc[r, c] = upd_str

    return copy

def yoy2_kpi_format(df_to_update):

    copy = df_to_update.copy()

    for c in range(0, df_to_update.shape[1]):
        if copy.columns[c] == 'acq_month':
            for r in range(0, df_to_update.shape[0]):
                upd_str = df_to_update.iloc[r, c]
                copy.iloc[r, c] = upd_str
        else:
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.0%}'.format(df_to_update.iloc[r, c])
                df_to_update.iloc[r, c]
                copy.iloc[r, c] = upd_str

    return copy

def act_kpi_format2(df_to_update):

    copy = df_to_update.copy()

    for c in range(0, df_to_update.shape[1]):
        if copy.columns[c] == 'acq_month':
            for r in range(0, df_to_update.shape[0]):
                upd_str = df_to_update.iloc[r, c]
                copy.iloc[r, c] = upd_str

        ## Sums and Counts first ##

        if copy.columns[c] == 'total_spend':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c]/1000000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'imps_1k':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c]/1000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'clicks_1k':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.2f}'.format(df_to_update.iloc[r, c]/1000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'total_installs':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.2f}'.format(df_to_update.iloc[r, c]/1000000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'user_count':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'm1_gmv':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c]/1000000) + ' MM'
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'm1_seller_gmv':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c]/1000000) + ' MM'
                copy.iloc[r, c] = upd_str

        if copy.columns[c] == 'm1_buyers':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'm1_listers':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'm1_sellers':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{:,.0f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str

        ## Cost per metrics ##

        if copy.columns[c] == 'cpm':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpc':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpi':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpu':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpb':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cpl':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cps':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '${:,.2f}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str

        ## Conversion Metrics ##

        if copy.columns[c] == 'ctr':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'ipc':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'upi':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'upc':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'buyer_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'lister_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'seller_act':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'm2_user_ret':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'm2_gmv_ret':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] ==  'cumm_m2_buyer_conv':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cumm_m2_lister_conv':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'cumm_m2_seller_conv':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'me':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str
        if copy.columns[c] == 'seller_scss':
            for r in range(0, df_to_update.shape[0]):
                upd_str = '{0:.2%}'.format(df_to_update.iloc[r, c])
                copy.iloc[r, c] = upd_str


    return copy
