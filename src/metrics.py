import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, root_mean_squared_log_error, r2_score


def safe_rmsle(estimator, X, y):
    y_pred = estimator.predict(X)
    y_pred_clipped = np.maximum(0, y_pred)
    return root_mean_squared_log_error(y, y_pred_clipped)


def evaluate(y_test, y_pred, return_dict=False):
    y_pred = np.maximum(0, y_pred)
    
    rmsle = root_mean_squared_log_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSLE (leaderboard): {rmsle:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    if return_dict:
        return {'RMSLE': rmsle, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
