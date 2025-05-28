from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_categorical_columns(data):
    """
    Returns a list of column names in the DataFrame that are categorical.
    Includes columns with dtype 'object' or 'category'.
    """
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    return cat_cols

def split_data(data, target, selected_columns=None, test_size=0.2, random_state=42):
    """
    Splits a DataFrame into X_train, X_test, y_train, y_test.
    Prints included features.
    
    Parameters:
    - data: input DataFrame.
    - target: target column name (string).
    - selected_columns: list of column names to include (if None, uses all except target).
    - test_size: fraction for test split.
    - random_state: random seed.
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    if selected_columns is None or len(selected_columns) == 0:
        used_features = [col for col in data.columns if col != target]
    else:
        used_features = selected_columns
    
    if len(used_features) == 0:
        raise ValueError("No features selected for splitting!")
    
    print("Included features:")
    for col in used_features:
        print(f"  - {col}")
    
    X = data[used_features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def feature_importance(pipeline, model_name, X_train):
    """
    Plots top_n feature importances from a fitted XGB pipeline.
    
    Parameters:
    - pipeline: your fitted sklearn Pipeline with XGBRegressor.
    - X_train: original (pre-encoded) training DataFrame.
    - top_n: number of top features to display.
    """
    # Get trained XGB model
    model = pipeline.named_steps[model_name]
    
    # Get preprocessor
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names after OneHotEncoder
    cat_features = preprocessor.transformers_[0][2]
    cat_ohe = preprocessor.transformers_[0][1]
    cat_names = cat_ohe.get_feature_names_out(cat_features)
    
    # Numeric features (passed through)
    num_features = [col for col in X_train.columns if col not in cat_features]
    
    # Combine all feature names
    all_feature_names = np.concatenate([cat_names, num_features])
    
    # Get importance scores
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
     

    return importance_df

def extract_ridge_feature_importance(ridge_pipeline, X_train, categorical_features, top_n=None, threshold=0.0):
    """
    Extracts feature importances from a fitted Ridge pipeline.

    Parameters:
    - ridge_pipeline: sklearn Pipeline with 'preprocessor' and 'ridge' steps.
    - X_train: original (pre-encoded) training DataFrame.
    - categorical_features: list of categorical feature names.
    - top_n: (optional) return only the top N features.
    - threshold: (optional) minimum absolute coefficient value to include.

    Returns:
    - DataFrame with Feature, Coefficient, and Absolute Importance, sorted by importance.
    """
    # Get preprocessor
    preprocessor = ridge_pipeline.named_steps['preprocessor']
    
    # Get OneHot feature names
    cat_ohe = preprocessor.named_transformers_['cat']
    cat_names = cat_ohe.get_feature_names_out(categorical_features)
    num_names = [col for col in X_train.columns if col not in categorical_features]
    all_feature_names = np.concatenate([cat_names, num_names])
    
    # Get coefficients
    ridge_model = ridge_pipeline.named_steps['ridge']
    coefs = ridge_model.coef_
    
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefs,
        'Importance': np.abs(coefs)
    }).sort_values(by='Importance', ascending=False)
    
    # Apply threshold filter
    importance_df = importance_df[importance_df['Importance'] >= threshold]
    
    # Apply top_n cutoff
    if top_n:
        importance_df = importance_df.head(top_n)
    
    return importance_df.reset_index(drop=True)

def compare_ridge_xgb_importance(
    ridge_pipeline, xgb_pipeline, X_train, 
    categorical_features, top_n=20
):
    # --- Get feature names ---
    preprocessor = ridge_pipeline.named_steps['preprocessor']
    cat_ohe = preprocessor.named_transformers_['cat']
    cat_names = cat_ohe.get_feature_names_out(categorical_features)
    num_names = [col for col in X_train.columns if col not in categorical_features]
    all_feature_names = np.concatenate([cat_names, num_names])
    
    # --- Ridge importance ---
    ridge_model = ridge_pipeline.named_steps['ridge']
    ridge_coefs = ridge_model.coef_
    ridge_importance = np.abs(ridge_coefs)
    
    ridge_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Ridge_Importance': ridge_importance
    }).set_index('Feature')
    
    # --- XGBoost importance ---
    xgb_model = xgb_pipeline.named_steps['xgb']
    xgb_importance = xgb_model.feature_importances_
    
    xgb_df = pd.DataFrame({
        'Feature': all_feature_names,
        'XGB_Importance': xgb_importance
    }).set_index('Feature')
    
    # --- Combine ---
    combined_df = ridge_df.join(xgb_df).fillna(0)
    combined_df['Ridge_Rank'] = combined_df['Ridge_Importance'].rank(ascending=False)
    combined_df['XGB_Rank'] = combined_df['XGB_Importance'].rank(ascending=False)
    
    # --- Select top features by average rank ---
    combined_df['Avg_Rank'] = (combined_df['Ridge_Rank'] + combined_df['XGB_Rank']) / 2
    top_features = combined_df.sort_values('Avg_Rank').head(top_n)
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    top_features.sort_values('Ridge_Importance', ascending=True)['Ridge_Importance'].plot.barh(ax=axes[0])
    axes[0].set_title('Top Ridge Feature Importance')
    axes[0].set_xlabel('Absolute Coefficient')
    
    top_features.sort_values('XGB_Importance', ascending=True)['XGB_Importance'].plot.barh(ax=axes[1], color='orange')
    axes[1].set_title('Top XGBoost Feature Importance')
    axes[1].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.show()
    
    return combined_df

def filter_features_from_original(feature_importance, original_dataframe, separator='_', importance_threshold=0.0):
    """
    Maps one-hot encoded pipeline feature names back to original DataFrame column names.
    
    Parameters:
    - feature_list: list of pipeline feature names (e.g., OneHot expanded).
    - original_columns: list of original DataFrame column names.
    - separator: character separating base and suffix (default '_').
    
    Returns:
    - unique list of matched original columns.
    """
    mapped_cols = set()
    important_features = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
    original_columns = set(original_dataframe.columns)
    
    for feat in important_features:
        if feat in original_columns:
            mapped_cols.add(feat)
        else:
            base = feat.split(separator)[0]
            if base in original_columns:
                mapped_cols.add(base)
    
    return original_dataframe[list(mapped_cols)]

def optimize_blend_weights(y_true, y_pred_xgb, y_pred_ridge):
    best_rmsle = float('inf')
    best_weight = None
    
    weights = np.arange(0, 1.05, 0.05)
    
    for w in weights:
        y_pred_blend = w * y_pred_xgb + (1 - w) * y_pred_ridge
        y_pred_blend = np.maximum(0, y_pred_blend)  # Clip negativos
        rmsle = root_mean_squared_log_error(y_true, y_pred_blend)
        
        print(f"Weight XGB: {w:.2f}, Weight Ridge: {1-w:.2f}, RMSLE: {rmsle:.5f}")
        
        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_weight = w
    
    print(f"\nBest blend: {best_weight:.2f} XGB + {1-best_weight:.2f} Ridge → RMSLE: {best_rmsle:.5f}")
    return best_weight, 1 - best_weight