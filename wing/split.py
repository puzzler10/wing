def split_dataset_with_sample(df_model, shuf, n_rows_sample, 
                              frac_train=0.7, frac_valid=0.15, df_keys=None): 
    """Splits out dataset into train, test and valid sets. The test set will have 
    (1-frac_train - frac_valid)*100% of the data. 
    This function takes samples of the data first. Total number of rows 
    across sets will be n_rows_sample.
    shuf: essentially if you want to shuffle your data. If you have a temporal element 
        don't do this
    n_rows_sample: number of rows to sample    
    frac_train: percentage of rows to have in training set 
    frac_valid: percentage of rows to have in validation set 
    df_keys: has date info used in shuf =False
    """
    # Take sample from dataframe 
    if shuf: 
        samp_model = sample_df(df_model, n_rows_sample, shuffle=True) # Get sample 
        train,valid,test = create_train_valid_test(samp_model, frac_train=frac_train,    # Split dataset 
                                                     frac_valid=frac_valid, temporal=False)
    else:  
        # Get sample
        # Adding ref_dt - used for sampling and splitting temporally
        date_str = df_keys.yr.map(str) + '-' + df_keys.mth.map(str).map(lambda x: x.zfill(2)) + '-01'
        df_keys['ref_dt'] = pd.to_datetime(date_str)
        df_model = df_model.merge(df_keys.loc[:,'ref_dt'].to_frame(), 
                                  left_index=True, right_index=True)
        samp_model = sample_df(df_model, n_rows_sample,  temporal=True, date_col='ref_dt')
        # Split datasaet
        train,valid,test = create_train_valid_test(samp_model, frac_train=frac_train, 
                                                 frac_valid=frac_valid, temporal=True, 
                                                 date_col='ref_dt', drop_date=True)
    return (train,valid,test)
 
 
 
 def split_X_y(train,valid,test,y_col):
    """get X,y train,test,valid"""
    def split(df, y_col): return df.drop(y_col, axis=1), df.loc[:,y_col]
    X_train,y_train = split(train,y_col)
    X_valid,y_valid = split(valid,y_col)
    X_test, y_test  = split(test,y_col)
    return (X_train,y_train,X_valid,y_valid,X_test,y_test)
    
    
def get_test_valid_model_score(model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, 
                         model_type='bin'):
    """ Gets log-loss of validation and test sets for a given classification algorithm model_name."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    model_type = 'bin'
    model_name = "rf_deep"
    if model_name == "rf_deep": 
        m = RandomForestClassifier(n_estimators = 70, max_features = 'sqrt', n_jobs = -1, 
                                  class_weight='balanced', min_samples_leaf=3)
    elif model_name == "rf_shallow":
        m = RandomForestClassifier(n_estimators = 30, max_features = 'auto', 
                                   n_jobs = -1, max_depth = 1)
    elif model_name == 'l1_logreg':     m = LogisticRegression(penalty='l1')
    elif model_name == 'l2_logreg':     m = LogisticRegression(penalty='l2')
    elif model_name == 'noReg_logreg':  m = LogisticRegression(C=99999999999)        
    elif model_name == 'decision_tree': m = DecisionTreeClassifier()
    m.fit(X_train,y_train)
    valid_probs,test_probs = m.predict_proba(X_valid),m.predict_proba(X_test)
    
    return (log_loss(y_valid, valid_probs),log_loss(y_test, test_probs))

    
    
def plot_test_valid_set_performance(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                    model_type='bin'): 
    """Compares performance (log-loss) of the valid and test sets for a number of classifiers. 
    This graph should be a straight line. If it isn't, it indicates that the validation 
    set is not representative of the test set."""
    import seaborn as sns 
    model_names = ['rf_deep', 'rf_shallow', 'l1_logreg', 
                   'l2_logreg','noReg_logreg', 'decision_tree']
    def empty_col(): return [None for o in range(len(model_names))]
    res = pd.DataFrame({'valid_score': empty_col(), 'test_score': empty_col()})
    for i,m_name in enumerate(model_names):
        res.iloc[i] = get_test_valid_model_score(m_name, X_train, y_train, X_valid, 
                                            y_valid, X_test, y_test, model_type=model_type)
    res = res.astype('float')
    res.index = model_names
    sns.regplot('valid_score', 'test_score', data=res)
    return res
    
    
def create_train_valid_test(df, frac_train=0.7, frac_valid=0.15,
                            temporal=False, date_col=None, shuffle=False,
                            drop_date=False, drop_other_cols=[]):
    """ Generates train, validation and test sets for data. Handles data with temporal components. 
    The test set will have (1 - frac_train - frac_valid) as a fraction of df 
    
    df: a Pandas dataframe
    frac_train: fraction of data to use in training set 
    frac_valid: fraction of data to use in validation set 
    temporal: does the data have a temporal aspect? Boolean
    date_col: the temporal column of df to order by. Required if temporal=True
    shuffle: shuffle the data in the training test sets (only valid for temporal=False)
    drop_date: drop the date column in the results or not 
    drop_other_cols: list with column names to drop 
    """
    if temporal and shuffle:      print("Shuffle = True is ignored for temporal data")
    if temporal and not date_col: raise ValueError("Need to pass in a value for date_col if temporal=True")
    if not temporal and date_col: print("Parameter for date_col ignored if temporal=False")
    frac_test = 1 - frac_train - frac_valid
    if temporal:
        # Sort the dataframe by the date column 
        inds = np.argsort(df[date_col])
        df = df.iloc[inds].copy()
    else: 
        if shuffle: 
            inds = np.random.permutation(df.shape[0])
            if type(df) == pd.DataFrame:   df = df.iloc[inds].copy()
            else:                          df = df[inds].copy()
    if drop_date: df.drop(date_col, axis=1, inplace=True)
    if drop_other_cols: df.drop(drop_other_cols, axis=1, inplace=True)
    train = df[0:int(df.shape[0] * frac_train)]
    valid = df[(train.shape[0]):(int(train.shape[0] + (df.shape[0] * frac_valid)))]
    test  = df[(train.shape[0] + valid.shape[0]):]
    return (train.copy(), valid.copy(), test.copy())
    
    
### For testing 
# numdays = 20
# base = datetime.datetime.today()
# date_list = [(base - datetime.timedelta(days=x)).date() for x in range(0, numdays)]
# y = [i for i in range(numdays)]
# x1 = [i**2 for i in range(numdays)]
# x2 = [np.sqrt(i) for i in range(numdays)]
# test_df = pd.DataFrame({
#     'y':y, 
#     'date_col': date_list, 
#     'x1': x1, 
#     'x2': x2
# })
# train,valid,test = create_train_valid_test(test_df, temporal=True, date_col = 'date_col')


def sample_df(df, n_rows, temporal=False, date_col=None, shuffle=False, seed=22):
    """Sample the most recent n_rows from df based on ordering from the date_col
    df: pandas dataframe 
    n_rows: how many rows to return
    date_col: what date column to use as ordering
    seed: seed used for random number generation"""
    np.random.seed(seed)
    if temporal and shuffle:      print("Shuffle = True is ignored for temporal data")
    if (type(df) == pd.DataFrame):
        if temporal and not date_col: raise ValueError("Need to pass in a value for date_col if temporal=True")
    else:  print("Parameter date_col ignored if df is not a data frame ")
    if not temporal and date_col: print("Parameter for date_col ignored if temporal=False")
    if temporal: 
        if type(df) == pd.DataFrame:   samp = df.sort_values(date_col, )[(-n_rows):]
        else:      
            df = np.array(df)   
            df.sort()
            df = list(df)
            samp = df[(-n_rows):]
    else: 
        if shuffle:
            if type(df) == pd.DataFrame:   samp = df.sample(n_rows)
            else: 
                inds = list(np.random.permutation(len(df)))[:n_rows]
                samp = [df[i] for i in inds]
        else:                        
            if type(df) == pd.DataFrame:   samp = df.iloc[:n_rows]
            else:                          samp = df[:n_rows]
    return samp.copy()



    
  