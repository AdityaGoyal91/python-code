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
