import os
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import umap
import xgboost as xgb
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import auc, classification_report, accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
def perform_eda(df, dataset_name):
    print(f"\n--- EDA for {dataset_name} ---\n")
    
    # 1. Overview of the dataset
    print("Shape of the dataset:", df.shape)
    print("First few rows of the dataset:")
    print(df.head())
    
    # 2. Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())
    
    # 3. Data types and summary statistics
    print("\nData types:")
    print(df.dtypes)
    print("\nStatistical Summary (numerical columns):")
    print(df.describe())
    
    # 4. Unique values in categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nUnique values in column '{col}':")
        print(df[col].value_counts())
    
    # 5. Data distribution for numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col} in {dataset_name}")
        plt.show()
    
    # 6. Bar plot for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Count of {col} in {dataset_name}")
        plt.xticks(rotation=45)
        plt.show()
    
    # 7. Correlation heatmap (for numerical columns)
    if len(numerical_cols) > 1:  # Only plot if there's more than one numerical column
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f"Correlation heatmap for numerical features in {dataset_name}")
        plt.show()
def split_data(df):
    if dataset_name=='Car_Insurance_Claim.csv':
        target_col='OUTCOME'
    elif dataset_name=='carclaims.csv':
        target_col='FraudFound'
    elif dataset_name=='insurance.csv':
        target_col='charges'
    elif dataset_name=='insurance_data.csv':
        target_col='CLAIM_STATUS'
    elif dataset_name=='test.csv':
        return
    else:
        target_col='Response'
    X = df.drop(target_col, axis=1)  # Replace 'target_column_name' with the name of your target column
    y = df[[target_col]]

# Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test

# Data Exploration
def plot_columns(columns, data, title, max_plots_per_figure=8, n_cols=4):
        n_rows = (n_cols + 3) // 4  # 4 plots per row, rounded up
        num_plots = len(columns)
        for start_idx in range(0, num_plots, max_plots_per_figure):
            cols_to_plot = columns[start_idx:start_idx + max_plots_per_figure]
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            plt.figure(figsize=(n_cols * 4, n_rows * 4))  # Adjust figure size dynamically
            for i, col in enumerate(cols_to_plot, 1):
                plt.subplot(n_rows, n_cols, i)
                sns.histplot(data=data, x=col, kde=True)
                plt.title(f'{col} Distribution')
                plt.xticks(rotation=45)
            plt.suptitle(title, y=1.02, fontsize=16)  # Add a title to the figure
            plt.tight_layout(pad=2, h_pad=2, w_pad=2)  # Adjust spacing
            plt.show()
# Data Exploration
def explore_data(df,dataset_name):
    #Initial trivial analysis
    print("Dataset Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicated Rows:", df.duplicated().sum())
    print("Handling missing values...")
    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype in ['float64', 'int64']:
                # Impute numerical features with median
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
                print(f"Filled missing values in numerical column '{column}' with median ({median_value}).")
            elif df[column].dtype == 'object':
                # Impute categorical features with mode
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
                print(f"Filled missing values in categorical column '{column}' with mode ('{mode_value}').")
    
    # 2. Removing Duplicated Rows
    initial_row_count = df.shape[0]
    df.drop_duplicates(inplace=True)
    final_row_count = df.shape[0]
    print(f"Removed {initial_row_count - final_row_count} duplicate rows.")
    '''
    total_instances = len(df)

    # Count of instances where feature = 0
    count_zero = len(df[df['fraud_bool'] == 0])

    # Count of instances where feature = 1
    count_one = len(df[df['fraud_bool'] == 1])

    # Calculate percentages
    percentage_zero = (count_zero / total_instances) * 100
    percentage_one = (count_one / total_instances) * 100

    print(f"Percentage of instances where feature = 0: {percentage_zero:.2f}%")
    print(f"Percentage of instances where feature = 1: {percentage_one:.2f}%")
    '''
    numerical_cols = df.select_dtypes(exclude=['object'])
    categorical_cols = df.select_dtypes(include=['object'])
    numerical_col_names = numerical_cols.columns.tolist()
    categorical_cols_names=categorical_cols.columns.tolist()
    # Separate 'fraud_bool' column
    if dataset_name=='Car_Insurance_Claim.csv':
        target_col='OUTCOME'
    elif dataset_name=='carclaims.csv':
        target_col='FraudFound'
    elif dataset_name=='insurance.csv':
        target_col='charges'
    elif dataset_name=='insurance_data.csv':
        target_col='CLAIM_STATUS'
    elif dataset_name=='test.csv':
        return
    else:
        target_col='Response'
    if target_col in numerical_col_names:
        numerical_col_names.remove(target_col)
        class_col = df[target_col]
    else:
        class_col = None    
    if target_col in categorical_cols_names:
        categorical_cols_names.remove(target_col)
        class_col=df[target_col]
    numerical_cols = df[numerical_col_names]
    categorical_cols=df[categorical_cols_names]    
    bool_cols = numerical_cols.columns[numerical_cols.nunique() == 2]  # columns with only two unique values (0 and 1)
    # Separate the binary columns
    binary_columns = df[bool_cols]
    for name in bool_cols:
        numerical_col_names.remove(name)
    numerical_cols = df[numerical_col_names]        
    #plot numerical columns
    plot_columns(numerical_cols.columns, numerical_cols, "Numerical Columns")
    #Plot categorical columns
    plot_columns(categorical_cols.columns, categorical_cols, "Categorical Columns", max_plots_per_figure=6)
    #plot binary columns
    plot_columns(binary_columns.columns, binary_columns, "Binary Columns", max_plots_per_figure=6)
    # Class distribution
    if class_col is not None:
        plt.figure(figsize=(8, 6))
        class_col.value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.show()
    if np.issubdtype(df[target_col].dtype, np.number):
        print("OK")
    else:
        df[target_col]=df[target_col].map({'No': 0, 'Yes': 1})
    return numerical_cols, categorical_cols,binary_columns, class_col
def clean_column_names(df):
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    return df

# Feature Selection
def feature_manipulation(X_train,X_test,y_train,y_test, numerical_cols, categorical_cols, binary_cols, class_col):
    #one hot encoding
    if dataset_name=='Car_Insurance_Claim.csv':
        target_col='OUTCOME'
    elif dataset_name=='carclaims.csv':
        target_col='FraudFound'
    elif dataset_name=='insurance.csv':
        target_col='charges'
    elif dataset_name=='insurance_data.csv':
        target_col='CLAIM_STATUS'
    elif dataset_name=='test.csv':
        return
    else:
        target_col='Response'
    X_train=pd.DataFrame(pd.get_dummies(X_train,prefix=categorical_cols.columns))
    X_test=pd.DataFrame(pd.get_dummies(X_test,prefix=categorical_cols.columns))
    categorical_cols=pd.DataFrame(pd.get_dummies(categorical_cols,prefix=categorical_cols.columns))
    #skewness check
     #log scaler: better for skewed features with wide range of values
    #min max scaler: better for uniform distributed features 
    #standard scaler: better for normally distributed
    #robust scaler: better for features with many outliers
   # numerical_cols.drop('month', axis=1, inplace=True)
    skewness = numerical_cols.apply(skew).sort_values(ascending=False)
    print("\nSkewness of Numerical Features:")
    print(skewness)
   # print(numerical_cols['device_fraud_count'].describe())
    skewed_features = skewness[skewness > 1].index.tolist()
    less_skewed_features = skewness[skewness < 1].index.tolist()
    non_skewed_features = skewness[skewness.isna()].index.tolist()  # implies constant feautre 
    for feature in non_skewed_features:
        X_train.drop(feature, axis=1, inplace=True)
        X_test.drop(feature,axis=1,inplace=True)
        numerical_cols.drop(feature, axis=1, inplace=True)
    #use log scaler for features with skewness >1 or <-1
    #problem may be here since some features contain negative values
    #will handle outliers too even though not really needed for tree classification methods 
    for feature in skewed_features:
        if (X_train[feature] > 0).all() and (X_test[feature]>0).all():  # Check if all values are positive
            X_train[feature] = np.log1p(X_train[feature])  # log(1 + x) to handle zero values
            X_test[feature]= np.log1p(X_test[feature])
        else:
            print(f"Feature '{feature}' contains non-positive values, skipping log transformation.")
            power_transformer = PowerTransformer(method='yeo-johnson')
            X_train[feature] = power_transformer.fit_transform(X_train[[feature]])
            X_test[feature] = power_transformer.transform(X_test[[feature]])
   # Assuming less_skewed_features is a list of column names
    min_max_scaler = MinMaxScaler()
    # Apply MinMaxScaler only to less_skewed_features, keeping other columns unchanged
    less_skewed_process = ColumnTransformer([('scaled', min_max_scaler, less_skewed_features)], remainder='passthrough')
    # Apply the transformation
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0  # Add missing columns with default value (e.g., 0)

    transformed_array = less_skewed_process.fit_transform(X_train)
    transformed_array_test = less_skewed_process.transform(X_test)
    # Convert back to DataFrame
    transformed_columns = less_skewed_features + [
        col for col in X_train.columns if col not in less_skewed_features
    ]
    X_train = pd.DataFrame(transformed_array, columns=transformed_columns)
    X_test = pd.DataFrame(transformed_array_test, columns=transformed_columns)
    #check skewness has been reduced 
    '''
    for feature in skewed_features:
        print(X_train[[feature]].skew())
    for feature in less_skewed_features:
        print(X_train[[feature]].skew())
    '''
    
    #check range of the numerical features not implemented yet 
    #multicollinearity numerical feature test
    col_corr=set()
    corr_matrix=X_train[numerical_cols.columns].corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if  (corr_matrix.iloc[i, j]) > 0.7:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)    
    print("Highly correlated columns (correlation > 0.7):", col_corr)
    #needed to change to int to test for correlation 
    y_train[target_col] = y_train[target_col].astype(int)
    y_test[target_col]=y_test[target_col].astype(int)
    final_features=set()
    corr=X_train.corrwith(y_train[target_col]).abs().sort_values(ascending=False)
    # Iterate over the index of the correlation series
    for feature in corr.index:
        print(f"{feature}: {corr[feature]}")
    f_scores, _ = f_classif(X_train[numerical_cols.columns], y_train[target_col])
    mutual_info = mutual_info_classif(X_train[numerical_cols.columns], y_train[target_col])
    # Create a DataFrame to rank features
    scores_df = pd.DataFrame({
        'Feature': numerical_cols.columns,
        'F-Score': f_scores,
        'Mutual Information': mutual_info
        }).sort_values(by='Mutual Information', ascending=False)

    print(scores_df)    
    best_mutual_info_cols = SelectKBest(mutual_info_classif, k=7)
    best_mutual_info_cols.fit(X_train[numerical_cols.columns], y_train[target_col])
    best_mutual_info_features = [X_train[numerical_cols.columns[best_mutual_info_cols.get_support()]]]
    print(best_mutual_info_features[0].columns)
    for column in best_mutual_info_features[0].columns:
        final_features.add(column)
    missing_cols = set(categorical_cols.columns) - set(X_train.columns)
    categorical_cols=categorical_cols.drop(columns=missing_cols)
    chi2_results = chi2(X_train[categorical_cols.columns], y_train[target_col])
# Create a DataFrame with the chi-squared results
    chi2_results_df = pd.DataFrame({
        'feature': categorical_cols.columns,  # Use the columns of categorical_cols directly
        'chi2': chi2_results[0],  # chi2 statistics
        'p_value': chi2_results[1]  # p-values
        })
    print(chi2_results_df)
    
# Plot the chi-squared results
    plt.figure(figsize=(16, 8))
    my_palette = sns.color_palette("husl", 2)
    sns.barplot(data=chi2_results_df.sort_values(by='chi2', ascending=False), 
            x="feature", y="chi2", palette=my_palette, alpha=.6)
# Customize labels and legend
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("chi2", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=5)
    plt.title("Chi-Squared Value by Categorical Feature", fontsize=14)
# Adjust the spacing between subplots
    plt.tight_layout()
# Show the plot
    plt.show()
    best_chi2_cols = SelectKBest(chi2, k=8)
    best_chi2_cols.fit(X_train[categorical_cols.columns], y_train[target_col])
    best_chi2_features = [X_train[categorical_cols.columns[best_chi2_cols.get_support()]]]
    best_chi2_features=list(best_chi2_features[0].columns)
    for column in best_chi2_features:
        final_features.add(column)
    extra = ExtraTreesClassifier(n_estimators=50, random_state=0)
    extra.fit(X_train, y_train)
    feature_sel_extra = SelectFromModel(extra, prefit=True)
    best_extra_features = [X_train.columns[(feature_sel_extra.get_support())]]
    best_extra_features = list(best_extra_features[0])
    binary_corr_with_target = binary_cols.corrwith(y_train[target_col])
    for index, value in binary_corr_with_target.items():
        if(value<0):
            final_features.add(index)        
    for column in best_extra_features:
        final_features.add(column)
    # below is a result of combination of  ouput from extraTreesClassif, correlation between features and class variable,
    #f-scores and mutual information, and chi-squared test. 
    final_features=list(final_features)
# Create a new DataFrame with only the selected features
    X_train = X_train[final_features]
# Find which numerical features are present in final_df
    numerical_in_final = [col for col in numerical_cols if col in final_features]
  #  print("Numerical features in final_df:", numerical_in_final)
# Find which categorical features are present in final_df
    categorical_in_final = [col for col in categorical_cols if col in final_features]
  #  print("Categorical features in final_df:", categorical_in_final)

# Find which binary features are present in final_df
    binary_in_final = [col for col in binary_cols if col in final_features]
  #  print("Binary features in final_df:", binary_in_final)
# Display the first few rows of the new DataFrame
 #  print(X_train.head())
    return X_train, X_test, y_train, y_test, numerical_in_final, categorical_in_final, binary_in_final

# Dimensionality Reduction I don't think we need this since we don't have too many features
def reduce_dimensions(df, numerical_cols):
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numerical_cols])

    # UMAP
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(df[numerical_cols])

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.scatter(pca_result[:, 0], pca_result[:, 1], c=df['fraud_bool'])
    ax1.set_title('PCA Results')

    ax2.scatter(umap_result[:, 0], umap_result[:, 1], c=df['fraud_bool'])
    ax2.set_title('UMAP Results')

    plt.show()

    return pca_result, umap_result

# Resampling
def resample_data(X, y):
    # Undersampling
    if isinstance(y, pd.DataFrame):
        y = y.values.ravel()  # Convert to 1D array
    elif isinstance(y, pd.Series):
        y = y.to_numpy()  # Ensure it's a NumPy array for compatibility
    nm = NearMiss()
    X_under, y_under = nm.fit_resample(X, y)

    # Oversampling
    smote = SMOTE()
    X_over, y_over = smote.fit_resample(X, y)

    # Visualize class distribution after resampling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    pd.Series(y_under).value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Class Distribution after Undersampling')

    pd.Series(y_over).value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title('Class Distribution after Oversampling')

    plt.show()
    # the model performs much better when oversampling is used so we will use oversampling 
    return X_over, y_over

# Time-based Train-Test Split
def time_split(df):
    train_df = df[df['month'].isin(range(6))]
    test_df = df[df['month'].isin([6, 7])]

    X_train = train_df.drop(['fraud_bool', 'month'], axis=1)
    y_train = train_df['fraud_bool'].to_frame('fraud_bool')
    X_test = test_df.drop(['fraud_bool', 'month'], axis=1)
    y_test = test_df['fraud_bool'].to_frame('fraud_bool')
    return X_train, X_test, y_train, y_test
def convert_data(X):
    for col in X.columns:
        if X[col].dtype == 'object':  # Only process object columns
            try:
                X[col] = X[col].astype(float)  # Direct conversion to int
            except ValueError:
                print(f"Column {col} contains non-numeric or NaN values, handling them.")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
    return X
# Model Training and Evaluation
#XGboost and LightGBM turned out to be the best I think
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        'XGBoost': xgb.XGBClassifier(),#I believe this will require conversion of our features from object to ints 
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'LightGBM': LGBMClassifier(),
      #  'Gradient Boosting': GradientBoostingClassifier()
    }

    results = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.show()

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.show()

    # Compare model performances
    results_df = pd.DataFrame(results).T
    plt.figure(figsize=(12, 6))
    results_df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return results
# Define base directory for the data lake
base_dir = 'data_lake'

# Define directories for raw and processed data
raw_data_dir = os.path.join(base_dir, 'raw')
processed_data_dir = os.path.join(base_dir, 'processed')

# Create directories if they do not exist
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)
# List of dataset files
dataset_files = glob(os.path.join(raw_data_dir, '*.csv'))

# Load datasets into a dictionary of DataFrames
dataframes = {os.path.basename(f): pd.read_csv(f) for f in dataset_files}
for name, df in dataframes.items():
    # Print columns to understand structure
    print(f"Dataset: {name}, Columns: {df.columns.tolist()}")
    print({df.columns.size})
    print({df.shape[0]})
#we are only considering vehicle and health insurance so 
#each dataset should only contain information pertaining to those 2 types
#but from our above exploration there is one dataset that contains multiple types of insurance
#so now we want to remove other insurance types
dataset_name = 'insurance_data.csv'
target_column = 'INSURANCE_TYPE'
allowed_values = ['Health', 'Motor']

# Check if the specified dataset is in the dataframes dictionary
if dataset_name in dataframes:
    df = dataframes[dataset_name]
    
    # Filter the DataFrame to keep only rows where INSURANCE_TYPE is 'health' or 'motor'
    initial_row_count = df.shape[0]  # Store the initial number of rows for verification
    df = df[df[target_column].isin(allowed_values)].copy()
    
    # Update the DataFrame in the dictionary
    dataframes[dataset_name] = df
    
    # Print the result to confirm rows were deleted
    final_row_count = df.shape[0]
    rows_deleted = initial_row_count - final_row_count
    print(f"Rows deleted in '{dataset_name}': {rows_deleted}")
    print(f"Remaining rows with '{target_column}' as 'health' or 'motor': {final_row_count}")
else:
    print(f"Dataset '{dataset_name}' not found.")
#now we have 8 datasets: 1 for car insurance fraud prediction, 1 for prediction if car policy will be claimed
#1 for predicting health insurance costs, 3 for predicting health and car fraud, and 1 for predicting if car insurance is needed
for dataset_name, df in dataframes.items():
   # perform_eda(df, dataset_name)
   numerical_cols, categorical_cols, binary_cols, class_col=explore_data(df,dataset_name)
   X_train, X_test, y_train, y_test = split_data(df)
   X_train, X_test, y_train, y_test, numerical_in_final, categorical_in_final, binary_in_final=feature_manipulation(X_train, X_test,y_train,y_test, numerical_cols,categorical_cols,binary_cols, class_col)
   X_train_resampled, y_train_resampled = resample_data(X_train, y_train)
   selected_features = X_train_resampled.columns
   X_test = X_test[selected_features]
   X_train_resampled=convert_data(X_train_resampled)
   X_test=convert_data(X_test)
   X_train = clean_column_names(X_train_resampled)
   X_test = clean_column_names(X_test)
   results = train_and_evaluate_models(X_train_resampled, X_test, y_train_resampled, y_test)

   
