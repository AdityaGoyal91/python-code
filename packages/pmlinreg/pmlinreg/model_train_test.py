def normalize_dataset_for_test(trainx, testx, train_stats):
    if set(trainx.columns) == set(testx.columns):
        norm_trainx = pd.DataFrame().reindex_like(trainx)
        norm_testx = pd.DataFrame().reindex_like(testx)

        for idx, datatype in enumerate(trainx.dtypes):
            if datatype == 'int64':
                col_name = trainx.columns[idx]
                temp_train_row = train_stats[train_stats['column_name']==col_name]
                print(temp_train_row)
                mean_train = temp_train_row.iloc[0]['mean']
                std_train = temp_train_row.iloc[0]['std']

                if std_train == 0:
                    norm_trainx[col_name] = trainx[col_name]
                    norm_testx[col_name] = testx[col_name]
                else:
                    norm_trainx[col_name] = (trainx[col_name] - mean_train)/std_train
                    norm_testx[col_name] = (testx[col_name] - mean_train)/std_train
            if datatype == 'float64':
                col_name = trainx.columns[idx]
                temp_train_row = train_stats[train_stats['column_name']==col_name]
                print(temp_train_row)
                mean_train = temp_train_row.iloc[0]['mean']
                std_train = temp_train_row.iloc[0]['std']

                if std_train == 0:
                    norm_trainx[col_name] = trainx[col_name]
                    norm_testx[col_name] = testx[col_name]
                else:
                    norm_trainx[col_name] = (trainx[col_name] - mean_train)/std_train
                    norm_testx[col_name] = (testx[col_name] - mean_train)/std_train

    elif set(trainx.columns) != set(testx.columns):
        print("Columns are not the same between train and test dataset, please make sure the columns match")

    return norm_trainx, norm_testx


def lm_coefs_rmse_df_output(trainx, trainy, testx, testy, test_df_to_append):
    lm = LinearRegression()
    lm.fit(trainx, trainy)
    coeff_output = {"Feature": trainx.columns, "estCoeff": lm.coef_, "Magnitude": abs(lm.coef_)}

    coeffs = pd.DataFrame(coeff_output).sort_values("Magnitude", ascending = 0)

    pred_df = pd.DataFrame(testy)

    pred_df.columns = ['act_y']

    for idx, val in enumerate(test_df_to_append.columns):
        pred_df[val] = test_df_to_append[val]

    pred_df = pred_df.reset_index(drop = True)

    pred_df['pred_y'] = pd.DataFrame(lm.predict(testx))

    pred_df['error_sq'] = (pred_df['pred_y'] - pred_df['act_y']) * (pred_df['pred_y'] - pred_df['act_y'])

    lm_rmse = math.sqrt((pred_df['error_sq'].sum()/pred_df['error_sq'].size))

    return coeffs, lm_rmse, pred_df, lm

def cat_variable_level_check(trainx, testx):

    train_level_cols = trainx.columns[(trainx.dtypes == 'object') | (trainx.dtypes == 'category')]

    test_level_cols = testx.columns[(testx.dtypes == 'object') | (trainx.dtypes == 'category')]

    if set(train_level_cols) == set(test_level_cols):
        print("Categorical variable columns match, checking for missing levels within the train data set")
        for idx, val in enumerate(test_level_cols):
            replace_list = np.setdiff1d(testx[val].unique(), trainx[val].unique())
            if all(replace_list == 0) == False:
                print("We are missing some levels either the test or train set for the column:", val, " Please fix before continuing.")
            else:
                print("All values match for the train and test categorical columns. Move to all column matching")
    else:
        print("The Categorical Variable set does not match, please check that the columns are the same")
    if set(trainx.columns) == set(testx.columns):
        print("All columns and levels match in the two data sets.")
    else:
        print("The columns in the train data set and test data set do not match, please go check.")



def rmse_decile(trainx, trainy, test, testycol, decile, col_for_rank):
    rmse_decile = pd.DataFrame(columns = ['Decile','Mean_value','Min_value','Max_value','Records','RMSE'])
    # train
    lm = LinearRegression()
    lm.fit(trainx, trainy)
    coeff_output = {"Feature": trainx.columns, "estCoeff": lm.coef_, "Magnitude": abs(lm.coef_)}

    coeffs = pd.DataFrame(coeff_output).sort_values("Magnitude", ascending = 0)

    test = test.sort_values(by=[col_for_rank])
    test = test.reset_index(drop = True)
    test['decile'] = pd.qcut(test.index, decile, labels=False)
    for i in test['decile'].unique():
        # get real spend (not normalized)
        temp_test = test[test['decile']==i]
        min_value = min(temp_test[col_for_rank])
        max_value = max(temp_test[col_for_rank])
        mean_value = sum(temp_test[col_for_rank])/len(temp_test[col_for_rank])

        #get test y and x
        temp_test_y = temp_test[testycol]
        temp_test_x= temp_test.loc[:,~temp_test.columns.isin([col_for_rank,'decile',testycol])]

        #calculate rmse
        pred_df = pd.DataFrame(temp_test_y).reset_index(drop=True)
        pred_df.columns = ['act_y']
        pred_df['pred_y'] = pd.DataFrame(lm.predict(temp_test_x))
        pred_df['error_sq'] = (pred_df['pred_y'] - pred_df['act_y']) * (pred_df['pred_y'] - pred_df['act_y'])
        lm_rmse = math.sqrt((pred_df['error_sq'].sum()/pred_df['error_sq'].size))
        rmse_decile = rmse_decile.append({'Decile':i+1,
                                          'Mean':mean_value,
                                          'Min':min_value,
                                          'Max':max_value,
                                          'Records':len(temp_test_x),
                                          'RMSE':lm_rmse}, ignore_index=True)
        rmse_decile = rmse_decile.sort_values(by=['Decile'])
    return rmse_decile


def static_lm_df_rmse(data, x_con_col, x_cat_col, y_col, output, start_prediction_date, end_prediction_date, date_col, not_normalized_col):

    bids_train = data[data[date_col] < start_prediction_date]
    bids_test = data[(data[date_col] >= start_prediction_date)
                     & (data[date_col] < end_prediction_date)]


    bids_train_y = bids_train[y_col]
    bids_test_y = bids_test[y_col]

    bids_test_output_col = bids_test[output]

    bids_train_cx = bids_train[x_con_col]
    bids_test_cx = bids_test[x_con_col]

    cat_variable_level_check(bids_train_cx, bids_test_cx)

    norm_bids_train_cx, norm_bids_test_cx = normalize_dataset(bids_train_cx, bids_test_cx)

    bids_catcon_train_x = norm_bids_train_cx.copy(deep=True)
    bids_catcon_test_x = norm_bids_test_cx.copy(deep=True)

    if len(x_cat_col) > 0:
        bids_addcat_train_x = bids_train[
            x_cat_col
        ]

        bids_addcat_test_x = bids_test[
            x_cat_col
        ]

        bid_addcat_x_model_train = pd.get_dummies(bids_addcat_train_x)
        bid_addcat_x_model_test = pd.get_dummies(bids_addcat_test_x)
        bid_addcat_x_model_test =  bid_addcat_x_model_test[bid_addcat_x_model_train.columns]

        bids_catcon_train_x = pd.concat([bids_catcon_train_x, bid_addcat_x_model_train],axis = 1)
        bids_catcon_test_x = pd.concat([bids_catcon_test_x, bid_addcat_x_model_test],axis = 1)

    bids_catcon_test = pd.concat([bids_catcon_test_x,bids_test_y],axis = 1)
    try:
        for nonncol in not_normalized_col:
            bids_catcon_test['true'+ nonncol]= bids_test[nonncol]
    except:
        pass

    temp_coeffs, temp_rmse, temp_reg_output_df, temp_model = lm_coefs_rmse_df_output(bids_catcon_train_x, bids_train_y, bids_catcon_test_x, bids_test_y, bids_test_output_col)

    temp_decile = rmse_decile(bids_catcon_train_x, bids_train_y, bids_catcon_test, y_col, 10)

    return temp_coeffs, temp_rmse, temp_reg_output_df, temp_decile, temp_model



def graph_daily_error(data_df, x_name, actual_y_name, pred_y_name, group_list, agg_func_list, graph_name, y_name):

    grouped_df = (
        data_df
            .groupby(
                group_list,
                as_index = False
            )
            .agg(agg_func_list)
    )

    grouped_df['gr_error_sq'] = (grouped_df[pred_y_name] - grouped_df[actual_y_name]) * (grouped_df[pred_y_name] - grouped_df[actual_y_name])
    grouped_rmse = math.sqrt((grouped_df['gr_error_sq'].sum()/grouped_df['gr_error_sq'].size))

    sns.set(font_scale = 1.5)

    fig, ax = pyplot.subplots(figsize = (14,6))

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.plot(grouped_df[x_name],grouped_df[actual_y_name])
    plt.plot(grouped_df[x_name],grouped_df[pred_y_name])

    ax.set_title(graph_name + ' Actual ' + y_name + ' vs Predicted ' + y_name, fontsize = 18)
    ax.legend(labels = ['actual','prediction'])
    ax.set_ylabel(y_name, fontsize = 16)
    ax.set_xlabel('Date', fontsize = 16)
    z.showplot(plt)
    return grouped_rmse, grouped_df



def static_load_test(model, data, x_con_col, x_cat_col, y_col, output, start_prediction_date, end_prediction_date, date_col, not_normalized_col):
    reg = model['model']
    bids_train = data[data[date_col] < start_prediction_date]
    bids_test = data[(data[date_col] >= start_prediction_date)
                     & (data[date_col] <= end_prediction_date)]
    print(bids_test[date_col].head())


    bids_train_y = bids_train[y_col]
    bids_test_y = bids_test[y_col]
    bids_test_output_col = bids_test[output]

    bids_train_cx = bids_train[x_con_col]
    bids_test_cx = bids_test[x_con_col]

    cat_variable_level_check(bids_train_cx, bids_test_cx)


    norm_bids_train_cx, norm_bids_test_cx = normalize_dataset_for_test(bids_train_cx, bids_test_cx, model.train_stats)

    bids_catcon_train_x = norm_bids_train_cx.copy(deep=True)
    bids_catcon_test_x = norm_bids_test_cx.copy(deep=True)

    if len(x_cat_col) > 0:
        bids_addcat_train_x = bids_train[
            x_cat_col
        ]

        bids_addcat_test_x = bids_test[
            x_cat_col
        ]

        bid_addcat_x_model_train = pd.get_dummies(bids_addcat_train_x)

        bid_addcat_x_model_test = pd.get_dummies(bids_addcat_test_x)


        bids_catcon_train_x = pd.concat([bids_catcon_train_x, bid_addcat_x_model_train],axis = 1)
        bids_catcon_test_x = pd.concat([bids_catcon_test_x, bid_addcat_x_model_test],axis = 1)

    bids_catcon_test = pd.concat([bids_catcon_test_x,bids_test_y],axis = 1)
    try:
        for nonncol in not_normalized_col:
            bids_catcon_test['true'+ nonncol]= bids_test[nonncol]
    except:
        pass

    temp_coeffs = model['coeff']

    pred_df = pd.DataFrame(bids_test_y)
    pred_df.columns = ['act_y']
    for idx, val in enumerate(bids_test_output_col.columns):
        pred_df[val] = bids_test_output_col[val]


    pred_df = pred_df.reset_index(drop = True)

    column_order = model.coeff['Feature']
    for col in column_order:
        if col in bids_catcon_test_x.columns:
            pass
        else:
            bids_catcon_test_x[col] = 0

    bids_catcon_test_x = bids_catcon_test_x[column_order]


    pred_df['pred_y'] = pd.DataFrame(reg.predict(bids_catcon_test_x))

    pred_df['error_sq'] = (pred_df['pred_y'] - pred_df['act_y']) * (pred_df['pred_y'] - pred_df['act_y'])

    temp_rmse = math.sqrt((pred_df['error_sq'].sum()/pred_df['error_sq'].size))


    return temp_coeffs, temp_rmse, pred_df
