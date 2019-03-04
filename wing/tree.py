import re, pandas as pd, numpy as np, matplotlib.pyplot as plt 
from sklearn.tree import export_graphviz
from sklearn.ensemble import forest
from scipy.cluster import hierarchy as hc
from concurrent.futures import ProcessPoolExecutor
from pdpbox import pdp

def draw_tree(t, df, size=10, ratio=0.6, precision=3, out_format='svg', out_path = './tree'):
    """ Draws a representation of a decision tree in IPython.
    # run in terminal first if on windows: conda install python-graphviz
    # todo: fix the terminal thing 
    Parameters:
    -----------
    t: The tree you wish to draw (a DecisionTreeRegressor, or  m.estimators_[i] if m is a RandomForestRegressor  )
    df: The data used to train the tree. This is used to get the names of the features.
    out_format: "inline" to plot to console, "ps" to save a postscript file, "png" to save png file
    out_path: if out_format is "ps" or "png" or "svg", saves output at this location (don't put extension at end)
    out_file: set to None to print to console, set to "xxx.dot" to save dot file as xxx.dot
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    s = re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s)
    if out_format == 'ps' or out_format == 'png' or out_format == 'svg':
        with open("tree.dot", "w") as text_file: text_file.write(s)
        command = f'dot -T{out_format} -Kneato -Goverlap=False tree.dot -o {out_path + "." + out_format}'
        print(command)
      #  !{command}
        os.remove('tree.dot')
    elif out_format == 'inline': 
        if out_path is not None: print("out_path ignored if out_format == 'inline'")
        IPython.display.display(graphviz.Source(s))  # display to console
    else: 
        raise ValueError("out_format should be one of 'inline', 'png' or 'ps'")
        
        
def plot_dendogram(df):
    """Dendogram to identify highly similar features in a dataframe """
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    fig = plt.figure(figsize=(16,10))
    dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
    plt.show()
   
   
def parallel_trees(m, fn, n_jobs=4):
    """# Linux only, or windows with python script and if name == __main__ shit
    
    ### use like below (for m being a rf model)
    if __name__ == '__main__':
        preds = np.stack([t.predict(X_valid) for t in m.estimators_])
        def get_preds(t): return t.predict(X_valid)
        %time preds = np.stack(parallel_trees(m, get_preds))
        np.mean(preds[:,0]), np.std(preds[:,0])
    """
    return list(ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))



def plot_pdp(feat, clusters=None, feat_name=None):
    """
    Might be a good idea to sample df for this analysis - generates lots of lines. 
    Maybe around 500 rows or so 
    m is a RandomForestRegression model from sklearn
    
    Examples
    plot_pdp('YearMade')
    plot_pdp('YearMade', clusters=5)

    # 2-D interaction plot 
    feats = ['saleElapsed', 'YearMade']
    p = pdp.pdp_interact(m, df, df.columns, feats)
    pdp.pdp_interact_plot(p, feats)

    # for 1-hot encoded columns 
    plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure')
    """
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, df, df.columns, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True,
                        cluster=clusters is not None,
                        n_cluster_centers=clusters)
                        

def rf_feat_importance(m, df):
    """Calculates matrix of feature importance for random forest m. df has the columns used to train m  """
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)

                       
def plot_fi(fi): 
    """Plots feature indepenence
    Example
    fi = rf_feat_importance(m, X_train)
    plot_fi(fi)""" 
    return fi.plot('cols', 'imp', 'barh', figsize=(12,14), color="#1f77b4", legend=False)


def set_rf_samples(n):
    """ Changes Scikit-learn's random forests to give each tree a random sample of
    n random rows. 
    Set oob_error in RandomForestRegressor to False if using this. 
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))


