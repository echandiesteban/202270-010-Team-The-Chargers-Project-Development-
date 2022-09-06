
# Import the os module
import os

# Get the current working directory
cwd = os.getcwd()
print(cwd)

# import pandas lib as pd
import pandas as pd
 
# read by default 1st sheet of an excel file
df = pd.read_excel('TestExcel.xlsx')
 
print(df)

print(df.label.unique())

# OUTPUT
#['Procedure Name' 'Action (What)' 'Action (Where)' 'Action (How)' 'Decide (What)' 'Decide (How)' 'Waiting (What)' 'Waiting (How)' 'Trigger (What)' 'Actor' 'Trigger (How)' 'Trigger (Where)']

print(df.isnull().sum())

data = df.dropna()

# Frecuencies

counts = data['label'].value_counts()
counts.plot(kind='bar', legend=False, grid=True, figsize=(8, 5))

import numpy as np

lens = data.text.str.len()
lens.hist(bins = np.arange(0,200,5))

#BERT Word Vectors

import spacy_sentence_bert


nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')

df['vector'] = df['text'].apply(lambda x: nlp(x).vector)


#split the data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(), df['label'].tolist(), test_size=0.1, random_state=690)

# train your choice of machine learning classifier

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=9, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

#Test Algorithm

Examples = ["thrust levers","release","the pilot","pf","PF","if the crosswind is at or below 20 knots and there is no tailwind :","in case of tailwind or if crosswind is greater than 20 knots:"]

label = ["Action (Where)", "Action (How)","Actor","Actor","Actor","Trigger (What)","Decide (What)"]

for Examples, label in zip(Examples, label):
  print(Examples,)
  print(f"True Label: {label}, Predicted Label: {clf.predict(nlp(Examples).vector.reshape(1, -1))[0]} \n")


