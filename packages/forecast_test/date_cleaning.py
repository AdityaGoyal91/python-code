def create_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create time-based features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['day_of_quarter'] = df.groupby(['year', 'quarter'])['day_of_month'].transform('cumcount') + 1
    
    # Create holiday flag using holidays library
    us_holidays = holidays.US()
    df['holiday_flag'] = df[date_col].isin(us_holidays)
    
    # Create working day flag using pandas
    df['working_day_flag'] = (df[date_col].dt.dayofweek < 5) & ~df['holiday_flag']
    
    return df
