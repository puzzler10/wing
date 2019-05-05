import matplotlib.pyplot as plt, pandas as pd, numpy as np

def set_plot_text_size(sml, med, big):
    """sml, med, big: text size of the small, medium and large text"""
    plt.rc('font', size=sml)          # controls default text sizes
    plt.rc('axes', titlesize=sml)     # fontsize of the axes title
    plt.rc('axes', labelsize=med)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=sml)    # fontsize of the tick labels
    plt.rc('legend', fontsize=sml)    # legend fontsize
    plt.rc('figure', titlesize=big)   # fontsize of the figure title
    
    
def plot_AvsEPred(y_true, y_pred, groups = 20, sample=True, model_name = "", save_plot = False, save_path = "", n_sample=20000):
    """
    Plot actuals vs expected values for a classifier. Useful for seeing model bias. 
    y_true: true y values, labels. Must be numeric 
    y_pred: y_predictions for y values. If for binary y_prediction, use the probability of the 1 class
    groups: how many bins to split values into? e.g. 20 will bin values every 5%, 100 will bin values every 1%
    """
    if n_sample > len(y_true): n_sample = len(y_true) 
    if len(y_true) != len(y_pred): raise ValueError("y_true and y_pred must be the same length")
    if type(y_true) == pd.Series: y_true = y_true.values
    if type(y_pred) == pd.Series:     y_pred = y_pred.values
    tmp_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    if sample:   tmp_df = tmp_df.sample(n_sample, random_state = 10).copy()  # useful for speed 
    y_true_ord = tmp_df.sort_values('y_pred', ascending=True)['y_true']
    y_pred_ord   = tmp_df.sort_values('y_pred', ascending=True)['y_pred']
 
    percentile_size = len(y_true_ord)/groups
    rank = np.linspace(1, len(y_true_ord), len(y_true_ord))
    perc_bands = np.ceil(rank/percentile_size)/groups
    
    y_true_avg = np.array(y_true_ord.groupby(perc_bands).mean())
    y_pred_avg = np.array(y_pred_ord.groupby(perc_bands).mean())
    x_axis = np.unique(perc_bands)
    
    ## Make plot 
    plt.subplots()
    t, = plt.plot(x_axis, y_true_avg, '-', color = 'blue', linewidth = 2, label = "t")
    p, = plt.plot(x_axis, y_pred_avg, '-', color = 'green', linewidth = 2, label = "p")
    plt.xlabel("y_predicted band", fontsize = 12)
    plt.ylabel("y_true", fontsize = 12)
    plt.title(("Actual vs y_predicted by y_predicted band\nModel: " + str(model_name)), fontsize = 14)
    plt.legend([t, p], ["Actual", "y_predicted"], fontsize = 12)
    if save_plot:
        if save_path is "":    plt.savefig(            "plot_AvsEy_pred_" + str(model_name) + ".png")
        else:                  plt.savefig(save_path + "plot_AvsEy_pred_" + str(model_name) + ".png")
        

        
def plot_gains(y_true, y_pred, sample=True, positive_target_only = False, n_sample=20000):
    """ 
    A gains/lift plot. 
    Set sample to True for a faster plot. Uses sample of the series to make the prediction"""
    if n_sample > len(y_true): n_sample = len(y_true) 
    if len(y_true) != len(y_pred): raise ValueError("y_true and y_pred must be the same length")
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
    
    
def get_palette(max_lvls, palette_type='categorical'): 
    """Returns a color palette for use when plotting bokeh"""
    from bokeh.palettes import Category20, Set3, BuPu, Reds
    if palette_type == 'categorical': 
        if   max_lvls == 2:    colors = BuPu[3][:2] 
        elif max_lvls <= 12:   colors = Set3[max_lvls]
        elif max_lvls <= 20:   colors = Category20[max_lvls]
        elif max_lvls <= 30:  
            colors = ["#4aa1ff","#d9ba00","#66008f","#85da78","#e278ff","#01d08c","#85007e","#b9d06b","#8482ff",
                      "#8e8800","#180043","#ff913a","#003572","#d74f00","#02dccd","#ff3272","#00723e","#ff6a5e",
                      "#0085a2","#91001f","#8fd4b7","#64004d","#6d5b00","#ffa6cc","#031700","#c9c5c5","#20001f",
                      "#016b68","#4d0006","#4a2d00"][0:max_lvls]
        elif max_lvls <= 40: 
            colors = ["#ff8b29","#7608c6","#00d161","#c000d0","#a3b800","#816fff","#187100","#e86bff","#a7d469",
                      "#ff46ce","#005e24","#6e007f","#5d7500","#0164cc","#a87500","#3c9cff","#782100","#49d4ff",
                      "#a30025","#86d4c7","#290047","#a5d29b","#9c0055","#123f00","#fdaaf4","#002817","#ff5961",
                      "#0099b4","#ff8160","#001338","#ffb091","#003d6f","#e8be96","#2a001f","#a8b1ff","#502600",
                      "#e6bacf","#6b0023","#016379","#ff749e"][0:max_lvls]
        elif max_lvls <= 50: 
            colors = ["#873d1f","#486ef8","#58c941","#a6009a","#63df71","#712fac","#c0d035",
                      "#0548c1","#579400","#c75ce2","#009624","#fb35a1","#01cf9c","#bd0073",
                      "#6bdc8c","#ff77c8","#008134","#c994ff","#657d00","#005bb7","#ff9120",
                      "#5baeff","#f36323","#02c2f0","#bd4e00","#018ccb","#aa8e00","#f4a4ff",
                      "#aad370","#ff64a2","#01956a","#ad0040","#00baaa","#ff7995","#235e2e",
                      "#faafe3","#595600","#005797","#f7bc61","#68447c","#b5cf93","#833765",
                      "#9eb07a","#823e3d","#55a1c9","#915700","#007765","#ff9ea5","#f7b992","#ffa696"][0:max_lvls]
        else: raise ValueError("Max_lvls for categorical palette type has to be between 2 and 50 inclusive")
    elif palette_type == 'sequential':
        if max_lvls <= 9 and max_lvls >= 3:   colors = Reds[max_lvls]
        elif max_lvls <= 20: 
            colors = ["#8da591","#1e0f13","#ffe6d7","#110d00","#ede0e3","#111c05","#e0ceb5","#002628","#dcb7bd",
                      "#07261c","#a9d5d7","#231d03","#5a818b","#351c18","#92797f","#1a2a31","#684d42","#2a535b","#41452e",
                      "#3d503d"][0:max_lvls]
        else: raise ValueError("Max_lvls for sequential palette type has to be between 3 and 20 inclusive")
    return colors 
    
    
def make_interactive_scatter_plot(df, x_col='coord_x', y_col='coord_y', color_col=None, 
                                  size_col=None, hover_cols=[], palette_type='categorical', color_bar=False, 
                                  plot_width=1200, plot_height=800, 
                                  max_lvls=20, plot_title='Title', plot_name='test.html'): 
    """
    Makes and saves interactive scatter plot to a file. 
        df: data frame for plotting 
        color_col: what column of df as the color label 
        x_col: what column of df to plot along x axis 
        y_col: what column of df to plot along y axis 
        hover_cols: what columns of df to come up when hovering over point 
        palette_type: either 'categorical' or 'sequential'
        plot_width: how wide is the output plot 
        plot_height: how high is the output plot 
        max_lvls: maximum number of categorical levels to plot. Chooses different palettes depending on the levels. Must be between 2 and 40 
        plot_title: Title of the output plot 
        plot_name: What name/path to save the output plot 
    """
    from bokeh.io import output_file, show, reset_output
    from bokeh.plotting import figure, show
    from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, BasicTicker
    from bokeh.transform import transform
    reset_output()
    source = ColumnDataSource(data=df)
    if palette_type == 'categorical': 
        if color_bar: raise ValueError("Must set palette_type='sequential' if specifying a color bar")
        n_levels = len(df[color_col].unique())
        factors = sorted(list(df[color_col].unique()))
        colors = get_palette(min(n_levels, max_lvls), palette_type=palette_type)
        color_mapper = CategoricalColorMapper(factors=factors, palette=colors)
    elif palette_type == 'sequential': 
        colors = get_palette(9, palette_type=palette_type)
        color_mapper = LinearColorMapper(low=df[color_col].min(), high=df[color_col].max(), palette=colors)
    hover_tooltips = [(o, "@" + o) for o in hover_cols]
    hover = HoverTool(tooltips=hover_tooltips)
    p = figure(plot_width=plot_width, plot_height=plot_height, 
               tools=[hover,'pan','box_zoom','wheel_zoom', 'reset'], title=plot_title)
    if size_col is None: p.circle(x_col, y_col, size=10,       source=source, fill_color=transform(color_col, color_mapper))
    else:                p.circle(x_col, y_col, size=size_col, source=source, fill_color=transform(color_col, color_mapper))
    if color_bar: 
        color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(), location=(0,0))
        p.add_layout(color_bar, 'right')
    output_file(plot_name)
    show(p)
    
    
    
    
    
    
    
    
    
    
    
    
  
