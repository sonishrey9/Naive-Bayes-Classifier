# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# %%
salary_data_test = pd.read_csv("SalaryData_Test.csv")
salary_data_train = pd.read_csv("SalaryData_Train.csv")


# %%
salary_data_test


# %%
salary_data_test.info()


# %%
cols = ['age', 'workclass', 'education', 'educationno', 'maritalstatus','occupation', 'relationship', 'race', 'sex', 'capitalgain','capitalloss', 'hoursperweek', 'native', 'Salary']
cols


# %%
salary_data_train.info()


# %%
print(salary_data_test.shape)
print(salary_data_train.shape)


# %%
# to perform classification first we'll be converting salary column to categorical feature of high and low
# new column is added to the dataframe
# for i in salary_data_test.Salary:
for i in salary_data_test.Salary:

    if i == " <=50K" :
        salary_data_test["Salary"] = salary_data_test["Salary"].replace([" <=50K"], "Low")
    elif i ==" >50K":
        salary_data_test["Salary"] = salary_data_test["Salary"].replace([" >50K"], "High")
        

for i in salary_data_train.Salary:

    if i == " <=50K" :
        salary_data_train["Salary"] = salary_data_train["Salary"].replace([" <=50K"], "Low")
    elif i ==" >50K":
        salary_data_train["Salary"] = salary_data_train["Salary"].replace([" >50K"], "High")


# %%
salary_data_test

# %% [markdown]
# ## **Exploratory Data Analysis**
# 
# ### **Both train and test datasets are same! so, we perform EDA on train Data**

# %%
## Here we will check the percentage of nan values present in each feature

## 1 -step make the list of features which has missing values

features_with_na=[features for features in salary_data_test.columns if salary_data_test[features].isnull().sum()>1] #list comprehension use

## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(salary_data_test[feature].isnull().mean(), 4),  ' % missing values')

# %% [markdown]
# ## No missing values in the given dataset
# %% [markdown]
# ## **Numerical Variables**

# %%

numerical_features = [feature for feature in salary_data_test.columns if salary_data_test[feature].dtypes != 'O'] # list comprehension feature that are not equal to object type

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
salary_data_test[numerical_features].head()

# %% [markdown]
# ## **Discrete Variables**

# %%
## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(salary_data_test[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
print(discrete_feature)

# %% [markdown]
# ## **Continous variable**

# %%
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature ]
print("Continuous feature Count {}".format(len(continuous_feature)))
print(continuous_feature)


# %%
categorical_features=[feature for feature in salary_data_test.columns if salary_data_test[feature].dtypes=='O']
categorical_features


# %%
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(salary_data_test[feature].unique())))

# %% [markdown]
# ### **Number of labels: cardinality**
# ### The number of labels within a categorical variable is known as cardinality. A high number of labels within a variable is ### known as high cardinality. High cardinality may pose some serious problems in the machine learning model. So, I will check ### for high cardinality

# %%
# check for cardinality in categorical variables

for var in categorical_features:
    
    print(var, ' contains ', len(salary_data_test[var].unique()), ' labels')


# %%
df = salary_data_test[categorical_features]
df.info()

# %% [markdown]
# ## **Label Encoding**

# %%

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()


# %%

for i in categorical_features: #label Encoding of test data
    salary_data_test[i] = label_encoder.fit_transform(salary_data_test[i])
    
    
for i in categorical_features: #label encoding of train data
    salary_data_train[i] = label_encoder.fit_transform(salary_data_train[i])


# %%
salary_data_test


# %%
salary_data_train


# %%
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

salary_data_train = scaler.fit_transform(salary_data_train)

salary_data_test = scaler.transform(salary_data_test)


# %%
salary_data_train = pd.DataFrame(salary_data_train, columns=[cols])
salary_data_train


# %%
salary_data_test = pd.DataFrame(salary_data_test, columns=[cols])
salary_data_test

# %% [markdown]
# ## **Model Training**

# %%
salary_data_test.shape, salary_data_train.shape


# %%
X = salary_data_train.drop(['Salary'], axis=1)

y = salary_data_train['Salary']


# %%
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# %%
# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)


# %%
y_pred = gnb.predict(X_test)

y_pred


# %%
from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# %%
y_pred_train = gnb.predict(X_train)

y_pred_train


# %%
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# %%
# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))


# %%
# check class distribution in test set

y_test.value_counts()


# %%
# check null accuracy score

null_accuracy = (7407/(7407+2362))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# %%
# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# %%
import seaborn as sns


# %%
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# %% [markdown]
# ## Classification accuracy

# %%
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# %%
# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))


# %%
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# %%
# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))


# %%
recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))


# %%
true_positive_rate = TP / float(TP + FN)


print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))


# %%
false_positive_rate = FP / float(FP + TN)


print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))


# %%
specificity = TN / (TN + FP)

print('Specificity : {0:0.4f}'.format(specificity))


# %%
# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = gnb.predict_proba(X_test)[0:10]

y_pred_prob


# %%
# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])

y_pred_prob_df


# %%
# print the first 10 predicted probabilities for class 1 - Probability of >50K

gnb.predict_proba(X_test)[0:10, 1]


# %%
# store the predicted probabilities for class 1 - Probability of >50K

y_pred1 = gnb.predict_proba(X_test)[:, 1]


# %%
import matplotlib.pyplot as plt


# %%
# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred1, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of salaries >50K')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')


# %%
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred1)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


# %%
# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(gnb, X_train, y_train, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

# %% [markdown]
# ## **k-Fold Cross Validation**

# %%
# Applying 10-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))


# %%
# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))


