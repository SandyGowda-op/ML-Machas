import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer, classification_report
from sklearn.impute import SimpleImputer
import joblib

# Column names for the NSL-KDD dataset
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# Load the dataset with specified column names
file_path = 'D:/Sandy Code/ML Hackathon/NSL_KDD.csv'  # Path to the uploaded NSL-KDD dataset

# Read the dataset and handle potential issues
data = pd.read_csv(file_path, names=col_names, low_memory=False)


# Define the mapping from detailed attack types to main categories
category_mapping = {
    'normal': 'normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L',
    'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'httptunnel': 'U2R',
    'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe'
}

# Apply the mapping to the dataset
data['label'] = data['label'].map(category_mapping)

# Drop rows with unmapped labels (if any)
data = data.dropna(subset=['label'])

# Convert appropriate columns to numeric, using coercion to handle errors
for col in col_names[:-1]:  # Exclude the label column
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=['protocol_type'])
data = data.dropna(subset=['flag'])
data = data.dropna(subset=['service'])

# Data preprocessing
# Separate features (X) and target (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Impute missing values for numerical data
'''num_imputer = SimpleImputer(strategy='mean')
X.loc[:, X.columns.difference(['protocol_type', 'service', 'flag'])] = num_imputer.fit_transform(
    X.loc[:, X.columns.difference(['protocol_type', 'service', 'flag'])]) '''

# Encode the target variable if it's categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Identify categorical columns
'''categorical_features = ["protocol_type", "service", "flag"] '''

# Preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        'num', StandardScaler()
    ])
print(data['service'])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

# Create a pipeline that includes preprocessing and the classifier
rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier(n_estimators=100, random_state=39))])

svm_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', SVC(kernel='linear', random_state=39))])

voting_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', VotingClassifier(estimators=[('rf', rf_clf.named_steps['classifier']),
                                                                    ('svc', svm_clf.named_steps['classifier'])], voting='hard'))])


# Performing Principal Component Analysis to decompose the 41 existing features to only 20 features
n_components = 20  # Set the number of principal components

# # Creating a pipeline that includes preprocessing, PCA, and the classifier
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X)
# X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# imputer = SimpleImputer(strategy='mean') 
# X = imputer.fit_transform(X)
# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X)

pca_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=n_components)),
    # ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_pca = pca_pipeline.fit(X)

# Train and evaluate the RandomForest model
def RandomForest():
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    print("RandomForest")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.5f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf, average='macro',zero_division=1):.5f}")
    print(f"Recall: {recall_score(y_test, y_pred_rf, average='macro'):.5f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_rf, average='macro'):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print()

    report = classification_report(y_test, y_pred_rf)
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap for Random Forest on NSL-KDD (5 Main Categories)")
    plt.show()


    # Save the RandomForest model
    joblib.dump(rf_clf, 'random_forest_model.pkl')

'''
# Train and evaluate the RandomForest model with PCA
def RandomForestPCA():
    

    rf_clf.fit(X_train_pca, y_train_pca)
    y_pred_rf_pca = pca.predict(X_test_pca)
    print("RandomForest with PCA")
    print(f"Accuracy: {accuracy_score(y_test_pca, y_pred_rf_pca):.5f}")
    print(f"Precision: {precision_score(y_test_pca, y_pred_rf_pca, average='macro', zero_division=1):.5f}")
    print(f"Recall: {recall_score(y_test_pca, y_pred_rf_pca, average='macro'):.5f}")
    print(f"F1 Score: {f1_score(y_test_pca, y_pred_rf_pca, average='macro'):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_pca, y_pred_rf_pca))
    print()

    report = classification_report(y_test_pca, y_pred_rf_pca)
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test_pca, y_pred_rf_pca), annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap for Random Forest with PCA on NSL-KDD (5 Main Categories)")
    plt.show()

    # Save the RandomForest model with PCA
    joblib.dump(rf_clf, 'random_forest_pca_model.pkl')
'''
def SVM():
    # Train and evaluate the SVM model
    svm_clf.fit(X_train, y_train)
    y_pred_svm = svm_clf.predict(X_test)
    print("Support Vector Machine")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.5f}")
    print(f"Precision: {precision_score(y_test, y_pred_svm, average='macro',zero_division=1):.5f}")
    print(f"Recall: {recall_score(y_test, y_pred_svm, average='macro',zero_division=1):.5f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_svm, average='macro'):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
    print()

    report = classification_report(y_test, y_pred_svm)
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap for Support Vector Machine on NSL-KDD (5 Main Categories)")
    plt.show()


    # Save the SVM model
    joblib.dump(svm_clf, 'svm_model.pkl')

'''
# Train and evaluate the RandomForest model with PCA
def RandomForestPCA():
    

    rf_clf.fit(X_train_pca, y_train_pca)
    y_pred_rf_pca = pca.predict(X_test_pca)
    print("RandomForest with PCA")
    print(f"Accuracy: {accuracy_score(y_test_pca, y_pred_rf_pca):.5f}")
    print(f"Precision: {precision_score(y_test_pca, y_pred_rf_pca, average='macro', zero_division=1):.5f}")
    print(f"Recall: {recall_score(y_test_pca, y_pred_rf_pca, average='macro'):.5f}")
    print(f"F1 Score: {f1_score(y_test_pca, y_pred_rf_pca, average='macro'):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_pca, y_pred_rf_pca))
    print()

    report = classification_report(y_test_pca, y_pred_rf_pca)
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test_pca, y_pred_rf_pca), annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap for Random Forest with PCA on NSL-KDD (5 Main Categories)")
    plt.show()

    # Save the RandomForest model with PCA
    joblib.dump(rf_clf, 'random_forest_pca_model.pkl')

'''
def Ensembling():
    # Train and evaluate the Voting Classifier
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    print("Voting Classifier (Ensembling):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_voting):.5f}")
    print(f"Precision: {precision_score(y_test, y_pred_voting, average='macro',zero_division=1):.5f}")
    print(f"Recall: {recall_score(y_test, y_pred_voting, average='macro',zero_division=1):.5f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_voting, average='macro'):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_voting))
    print()

    report = classification_report(y_test, y_pred_voting)
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred_voting), annot=True, fmt="d", cmap="YlGnBu", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap for Voting Classifier Ensembling on NSL-KDD (5 Main Categories)")
    plt.show()


    # Save the Voting Classifier
    joblib.dump(voting_clf, 'voting_classifier_model.pkl')

n = int(input("Choose your Machine Learning Model:\n1. Random Forest Classifier\n2. Support Vector Machine\n3. Voting Classifier\n4. All Models\n"))
if n == 1:
    RandomForest()
    '''ch = int(input("1. Check the results for random forest after PCA Decomposition\n2. Continue\n"))
    if ch == 1:
        RandomForestPCA()
    elif ch == 2:
        pass
    else:
        print("Invalid Input")'''
elif n==2:
    SVM()
elif n==3:
    Ensembling()
elif n==4:
    RandomForest()
    SVM()
    Ensembling()
else:
    print("Invalid Input")

