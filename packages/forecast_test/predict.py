def predict_future(df: pd.DataFrame, date_col: str, train_range: Tuple[str, str], pred_range: Tuple[str, str], targets: List[str]) -> pd.DataFrame:
    df = create_features(df, date_col)

    train_start, train_end = pd.to_datetime(train_range[0]), pd.to_datetime(train_range[1])
    pred_start, pred_end = pd.to_datetime(pred_range[0]), pd.to_datetime(pred_range[1])

    train_df = df[(df[date_col] >= train_start) & (df[date_col] <= train_end)]
    pred_df = df[(df[date_col] >= pred_start) & (df[date_col] <= pred_end)]

    base_features = ['day_of_week', 'day_of_month', 'day_of_quarter', 'spend', 'holiday_flag', 'working_day_flag', 'year', 'month', 'quarter']
    
    for target in targets:
        features = base_features + [f'predicted_{t}' for t in targets[:targets.index(target)]]
        model = PredictionModel(features)
        X, y = model.prepare_data(train_df, target)
        model.train_models(X, y)
        model.select_best_model(X, y)
        
        pred_features = pred_df[features].values
        predictions = model.predict(pred_features)
        pred_df[f'predicted_{target}'] = predictions
        
        # Add the prediction as a feature for the next target
        df[f'predicted_{target}'] = df[target]
        df.loc[(df[date_col] >= pred_start) & (df[date_col] <= pred_end), f'predicted_{target}'] = predictions

    return pred_df[[date_col] + [f'predicted_{target}' for target in targets]]
