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
