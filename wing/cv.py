def analyse_cv_results(searcher_result):
    """
    Returns best estimator, dataframe with cv results, and std of results over each variable.
    searcher_result: the output from RandomizedSearchCV (might work for GridSearcCV)
    
    """
    # Identify best model for easy returning 
    bestmodel = searcher_result.best_estimator_
    # Get predictor dataframe with scores for analysis 
    tmp_df = pd.DataFrame(searcher_result.cv_results_['params'])
    parameters_df = pd.concat([
        pd.Series(searcher_result.cv_results_['mean_test_score']), tmp_df
    ],axis=1, ignore_index=True)
    parameters_df.columns = ['test_score'] + list(tmp_df.columns)
    # Look at the standard deviation in test scores across each variable.
    # Gives indication of if variable is important or not
    # (probably doesn't work for parameters with a cts distribution)
    stds = []
    cnames = [o for o in parameters_df.columns if o != 'test_score']
    for cname in cnames: 
        stds.append(parameters_df.groupby(cname).agg('mean').test_score.std())
    std_df = pd.DataFrame({'colname':cnames, 
                          'score_std_across_lvls': stds})
    std_df = std_df[~std_df.score_std_across_lvls.isnull()]
    return (bestmodel, parameters_df, std_df)
