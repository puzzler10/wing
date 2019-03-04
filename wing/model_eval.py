import math
from numpy import sqrt
from sklearn.metrics import mean_squared_error, r2_score,f1_score, roc_auc_score, log_loss

def print_score(y_true_train, y_preds_train, y_true_valid, y_preds_valid, y_probs_train=None, y_probs_valid=None,
                model_type='reg', print_res=True, return_res=False):
    """
    Print model evaluation statistics for train and validation sets. 
    For regression (type='reg') will print RMSE and Rsquared. For binary classification ('bin') will print f1 score, 
    log loss and ROC AUC. 
    
    Parameters: 
    y_true_train,y_true_valid: true values for y over train and valid sets 
    y_preds_train,y_preds_valid: model predictions for y over train and valid sets (e.g. m.predict(X_train))
    y_probs_train,y_probs_valid: applies only to binary classification (type='bin'). Model probablities for each class. 
        (e.g. m.predict_proba(X_train))
    type: one of 'reg'(regression) or 'bin' (binary classification)
    set print_res to True to print results and names 
    set return_res to True to return names and results in a tuple (names, res) 
    
    Example: 
    y_preds_train,y_preds_valid = m.predict(X_train),      m.predict(X_valid)
    y_probs_train,y_probs_valid = m.predict_proba(X_train),m.predict_proba(X_valid)
    print_score(y_true_train, y_preds_train, y_true_valid, y_preds_valid, y_probs_train, y_probs_train, type='bin')
    """
    if model_type == 'reg': 
        names = ["RMSE Train: ", "RMSE Valid: ", "RSquared Train: ","RSquared Valid: " ]
        res = [sqrt(mean_squared_error(y_true_train, y_preds_train)), sqrt(mean_squared_error(y_true_valid, y_preds_valid)),
               r2_score(y_true_train, y_preds_train), r2_score(y_true_valid, y_preds_valid)]
    elif model_type == 'bin':
        # Some metrics need probs (float probs of each class), some need preds (1 or 0)
        #y_true_train,y_true_valid = y_true_train.astype(int).copy(),y_true_valid.astype(int).copy()
        #y_preds_train,y_preds_valid = y_preds_train.astype(int).copy(),y_preds_valid.astype(int).copy()
        names = ["Log Loss Train: ", "Log Loss Valid: ", "F1 Train: ", "F1 Valid: ", "AUC Train: ","AUC Valid: " ]
        res = [     log_loss(y_true_train, y_probs_train),      log_loss(y_true_valid, y_probs_valid),
                    f1_score(y_true_train, y_preds_train),      f1_score(y_true_valid, y_preds_valid),
               roc_auc_score(y_true_train, y_preds_train), roc_auc_score(y_true_valid, y_preds_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_); names.append("OOB Score: ")    
    if print_res: 
        for name,score in zip(names,res): print (f'{name}{score}')
    if return_res:  return names, res

    
def rmse(x,y): return math.sqrt(((x-y)**2).mean())


def predict_and_print_score(m, X_train, X_valid, y_train, y_valid, type='reg', print_res=True, return_res=False):
    """Print RMSE and R**2 for a model on train and validation sets. 
    m is a model. type is one of 'reg'(regression) or 'bin' (binary classification)
    set return_res to True to return results and names
    set print_res to True to print results and names     """
    if type == 'reg': 
        X_train_preds,X_valid_preds = m.predict(X_train),m.predict(X_valid)
        names = ["RMSE Train: ", "RMSE Valid: ", "RSquared Train: ","RSquared Valid: " ]
        res = [sqrt(mean_squared_error(y_train, X_train_preds)), sqrt(mean_squared_error(y_valid, X_valid_preds)),
               r2_score(y_train, X_train_preds), r2_score(y_valid, X_valid_preds)]
    elif type == 'bin':
        # Some metrics need probs (float probs of each class), some need preds (1 or 0)
        X_train_preds,X_valid_preds = m.predict(X_train).astype(int),m.predict(X_valid).astype(int)
        X_train_probs,X_valid_probs = m.predict_proba(X_train),m.predict_proba(X_valid)
        y_train,y_valid = y_train.astype(int).copy(),y_valid.astype(int).copy()
        names = ["Log Loss Train: ", "Log Loss Valid: ", "F1 Train: ", "F1 Valid: ", 
        "AUC Train: ","AUC Valid: " ]
        res = [log_loss(y_train, X_train_probs), log_loss(y_valid, X_valid_probs),
               f1_score(y_train, X_train_preds), f1_score(y_valid, X_valid_preds),
               roc_auc_score(y_train, X_train_preds), roc_auc_score(y_valid, X_valid_preds)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_); names.append("OOB Score: ")    
    if print_res: 
        for name,score in zip(names,res): print (f'{name}{score}')
    if return_res:  return names, res
 
