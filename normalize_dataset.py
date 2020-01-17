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
