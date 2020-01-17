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
