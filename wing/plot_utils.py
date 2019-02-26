def set_plot_text_size(sml, med, big):
    """sml, med, big: text size of the small, medium and large text"""
    plt.rc('font', size=sml)          # controls default text sizes
    plt.rc('axes', titlesize=sml)     # fontsize of the axes title
    plt.rc('axes', labelsize=med)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('legend', fontsize=sml)    # legend fontsize
    plt.rc('figure', titlesize=big)   # fontsize of the figure title
    
    
def plot_AvsEPred(target, pred, groups = 20, sample=True, model_name = "", save_plot = False, save_path = "", n_sample=20000):
    """
    Plot actuals vs expected values for a classifier. Useful for seeing model bias. 
    target: true y values, labels. Must be numeric 
    pred: predictions for y values. If for binary prediction, use the probability of the 1 class
    groups: how many bins to split values into? e.g. 20 will bin values every 5%, 100 will bin values every 1%
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    if type(target) == pd.Series: target = target.values
    if type(pred) == pd.Series:     pred = pred.values
    tmp_df = pd.DataFrame({'target': target, 'pred': pred})
    if sample:   tmp_df = tmp_df.sample(n_sample, random_state = 10).copy()  # useful for speed 
    target_ord = tmp_df.sort_values('pred', ascending=True)['target']
    pred_ord   = tmp_df.sort_values('pred', ascending=True)['pred']
    
    percentile_size = len(target_ord)/groups
    rank = np.linspace(1, len(target_ord), len(target_ord))
    perc_bands = np.ceil(rank/percentile_size)/groups
    
    target_avg = np.array(target_ord.groupby(perc_bands).mean())
    pred_avg = np.array(pred_ord.groupby(perc_bands).mean())
    x_axis = np.unique(perc_bands)
    
    ## Make plot 
    plt.subplots()
    t, = plt.plot(x_axis, target_avg, '-', color = 'blue', linewidth = 2, label = "t")
    p, = plt.plot(x_axis, pred_avg, '-', color = 'green', linewidth = 2, label = "p")
    plt.xlabel("Predicted band", fontsize = 12)
    plt.ylabel("Target", fontsize = 12)
    plt.title(("Actual vs predicted by predicted band\nModel: " + str(model_name)), fontsize = 14)
    plt.legend([t, p], ["Actual", "Predicted"], fontsize = 12)
    if save_plot:
        if save_path is "":    plt.savefig(            "plot_AvsEPred_" + str(model_name) + ".png")
        else:                  plt.savefig(save_path + "plot_AvsEPred_" + str(model_name) + ".png")
        

        
def plot_gains(y_true, y_pred, sample=True, positive_target_only = False, n_sample=20000):
    """ 
    A gains/lift plot. 
    Set sample to True for a faster plot. Uses sample of the series to make the prediction"""
    import matplotlib.pyplot as plt
    if type(y_true) == pd.Series:     y_true = y_true.values
    if type(y_pred) == pd.Series:     y_pred = y_pred.values
    if positive_target_only: y_true, y_pred = y_pred[y_true > 0], y_true[y_true > 0]
    y_true,y_pred = y_true[~np.isnan(y_true)],y_pred[~np.isnan(y_true)]     # remove entries with missing y_true obs
    tmp_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    if sample: tmp_df = tmp_df.sample(n_sample, random_state = 10).copy()  # useful for speed
    def scale_cumsum_sort_and_add0(df, sort_col=''): 
        y1 = df.sort_values(sort_col, ascending=False).y_true.values
        return np.append(0, np.cumsum(y1) / np.sum(y1))
    y_max_gains   = scale_cumsum_sort_and_add0(tmp_df, 'y_true')
    y_model_gains = scale_cumsum_sort_and_add0(tmp_df, 'y_pred')
    #### Make plot 
    plt.subplots()
    x = np.linspace(0,1,len(y_max_gains))
    # max, model and random gains
    t, = plt.plot(x, y_model_gains,   '-', color = 'blue',  linewidth = 2, label = "t")
    p, = plt.plot(x, y_max_gains,     '-', color = 'green', linewidth = 2, label = "p")
    r, = plt.plot(x, x,               '-', color = 'grey',  linewidth = 2, label = "r")
    plt.xlabel("Cumulative proportion of population", fontsize = 12)
    plt.ylabel("Gains", fontsize = 12)
    plt.title("Gains chart", fontsize = 14)
    plt.legend([t, p, r], ["Model Gains", "Theoretical Max Gains", "Random Gains"], fontsize = 12)
    #### Calc percentage of theoretical max gains 
    def calc_area_simpsons_rule(x,y):
        h = (x[2] - x[0]) / 2   # assume x is evenly spaced 
        f_a, f_half, f_b = y[:-2:2], y[1:-1:2], y[2::2]
        return sum((h / 3) * (f_a + 4 * f_half + f_b))
    # We minus 0.5 because we calculate the area between the random gains curve 
    # and the model/max curves. The random gains curve has an area of 0.5 by definition
    area_max   = calc_area_simpsons_rule(x, y_max_gains)   - 0.5
    area_model = calc_area_simpsons_rule(x, y_model_gains) - 0.5
    return area_model / area_max
