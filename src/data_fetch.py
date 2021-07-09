# importing the dataset
import pickle
from typing import Dict, Any

import numpy as numpy
import pandas
import numpy
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pandas.read_csv("https://raw.githubusercontent.com/hoshangk/machine_learning_model_using_flask_web_framework"
                     "/master/adult.csv")
# print(df.head())
# print(df.columns)
# print(df.mode())

df = df.drop(['fnlwgt', 'educational-num'], axis=1)
# print(df.columns)

col_names = df.columns

for column in col_names:
    df = df.replace("?", numpy.NAN)
# print(df.mode())

df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

# print(df.mode())
# print(df.dtypes)
# print(df['marital-status'].unique())
# df["column1"].replace({"a": "x", "b": "y"}, inplace=True)
df["marital-status"].replace(
    {" Divorced": "not married", " Never-married": "not married", " Separated": "not married",
     " Married-spouse-absent": "married",
     " Married-civ-spouse": "married", " Married-AF-spouse": "married", " Widowed": "married"}, inplace=True)

# df['marital-status'].replace(['Divorced', 'Married-AF-spouse',
#                               'Married-civ-spouse', 'Married-spouse-absent',
#                               'Never-married', 'Separated', 'Widowed'],
#                              ['divorced', 'married', 'married', 'married',
#                               'not married', 'not married', 'not married'], inplace=True)

# print(df['marital-status'].unique())

# df1 = pandas.DataFrame({"column1": ["a", "b", "a"]})
# print(df1)
# df1["column1"].replace({"a": "x", "b": "y"}, inplace=True)
# print(df1)
#
# df["marital-status"].replace({" Divorced": "x"}, inplace=True)
# print(df['marital-status'].unique())


category_col = df.select_dtypes("object")
# print(category_col)

label_encoder = preprocessing.LabelEncoder()

mapping_dict: Dict[Any, Dict[Any, Any]] = {}
for column in category_col:
    df[column] = label_encoder.fit_transform(df[column])
    label_encoder_mapping_classes = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    # print(label_encoder_mapping_classes)
    mapping_dict[column] = label_encoder_mapping_classes
# print("*****************************************************************************8")
# print(mapping_dict)

# df1 = pandas.DataFrame([('bird', 2, 2),
#                         ('mammal', 4, numpy.nan),
#                         ('arthropod', 8, 0),
#                         ('bird', 2, numpy.nan)], index=('falcon', 'horse', 'spider', 'ostrich'),
#                        columns=('species', 'legs', 'wings'))
# print(df1)
# print(df1.mode(axis='rows', numeric_only=True))

X = df.values[:, 0:12]  # all rows for columns till 0 to 11
Y = df.values[:, 12]  # all rows for column 12 - The Target Class/variable

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

dt_clf_gini = DecisionTreeClassifier(criterion="gini",
                                     random_state=100,
                                     max_depth=5,
                                     min_samples_leaf=5)

dt_clf_gini.fit(X_train, y_train)
y_predicted_gini = dt_clf_gini.predict(X_test)

print(f"Predicted Gini Index:{y_predicted_gini}")

print("Decision Tree using Gini Index\nAccuracy is ",
      accuracy_score(y_test, y_predicted_gini) * 100)

# Saving the model
# joblib.dump(dt_clf_gini, 'adults_dtree.pkl')
# pickle.dump(dt_clf_gini, '..//adults1_dtree.pkl')
# save data to a file
with open('..//myfile.pickle', 'wb') as fout:
    pickle.dump(dt_clf_gini, fout)
