# Importing necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Loading the dataset from a CSV file
crops = pd.read_csv("soil_measures.csv")

# Creating a DataFrame from the loaded data
crops_df = pd.DataFrame(crops)
print(crops_df.head())  # Display the first few rows of the DataFrame

# Checking for missing values in the dataset
print(crops_df.isna().sum().sort_values())

# Displaying the unique values of the 'crop' column (target variable)
print(crops_df.crop.unique())  # 'crop' is a multi-label feature

# Separating features (X) and target (y)
X = crops_df.drop("crop", axis=1).values
y = crops["crop"].values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing dictionaries to store performance metrics
best_predictive_feature = {}
feature_accuracy = {}
feature_performance = {}

# Iterating over each feature to evaluate its predictive power
for feature in ["N", "P", "K", "ph"]:
    # Initializing a Logistic Regression model with specified parameters
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)

    # Getting the index of the current feature
    feature_index = crops_df.columns.get_loc(feature)

    # Training the model using only the current feature
    logreg.fit(X_train[:, feature_index].reshape(-1, 1), y_train)

    # Predicting the target values for the test set
    y_pred = logreg.predict(X_test[:, feature_index].reshape(-1, 1))

    # Calculating the balanced accuracy score
    accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
    feature_accuracy[feature] = accuracy

    # Calculating the F1-score
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    feature_performance[feature] = f1

    # Printing the accuracy and F1-score for the current feature
    print(f"Accuracy for {feature}: {feature_accuracy}")
    print(f"F1-score for {feature}: {feature_performance}")

# Identifying the feature with the highest balanced accuracy
best_feature = max(feature_accuracy, key=feature_accuracy.get)

# Storing the best predictive feature and its accuracy in the dictionary
best_predictive_feature = {best_feature: feature_accuracy[best_feature]}

# Printing the best predictive feature
print(f"Best predictive feature:", best_predictive_feature)