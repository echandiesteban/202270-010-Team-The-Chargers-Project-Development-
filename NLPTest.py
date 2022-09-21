
# Import 
import os
import numpy as np
import seaborn as sns
import pandas as pd
import spacy_sentence_bert


# Get the current working directory

cwd = os.getcwd()
print(cwd)

# read by default 1st sheet of an excel file
df = pd.read_excel('TestExcel5.xlsx')

#from nltk.corpus import stopwords
#stop = stopwords.words('english')
#df["Text_wo"] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df["Text_wo"]= df["Text_wo"].str.replace(r'\W'," ")


print(df)

print(df.label.unique())

# OUTPUT
#['Procedure Name' 'Action (What)' 'Action (Where)' 'Action (How)' 'Decide (What)' 'Decide (How)' 'Waiting (What)' 'Waiting (How)' 'Trigger (What)' 'Actor' 'Trigger (How)' 'Trigger (Where)']

print(df.isnull().sum())

data = df.dropna()

# Frecuencies
counts = data['label2'].value_counts()
counts.plot(kind='bar', legend=False, grid=True, figsize=(8, 5))

counts = data['label'].value_counts()
counts.plot(kind='bar', legend=False, grid=True, figsize=(8, 5))

lens = data.text.str.len()
lens.hist(bins = np.arange(0,200,5))

#BERT Word Vectors

nlp = spacy_sentence_bert.load_model('en_stsb_bert_large')
#nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')

df['vector'] = df['text'].apply(lambda x: nlp(x).vector)


#**************LABEL***************
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

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)


#**************LABEL2***************
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(), df['label2'].tolist(), test_size=0.1, random_state=690)

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

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)

#Test Algorithm

Examples = ['takeoff', 'if the crosswind' ,'Pilot']

label = ['Action (What)', 'Decide (What)' ,'Actor']

for Examples, label in zip(Examples, label):
  print(Examples,)
  print(f"True Label: {label}, Predicted Label: {clf.predict(nlp(Examples).vector.reshape(1, -1))[0]} \n")


# Visualizations

from sklearn.manifold import TSNE

X = list(df["vector"])

X_embedded = TSNE(n_components=2).fit_transform(X)

df_embeddings = pd.DataFrame(X_embedded)
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})


df2 = pd.merge(df, df_embeddings, left_index=True, right_index=True)

groups = df2.groupby("label")

for name, group in groups:
    plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name,markersize=3)
plt.legend(loc='best', bbox_to_anchor=(0.5,-0.1))


groups2 = df2.groupby("label2")

for name, group in groups2:
    plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name,markersize=3)
plt.legend(loc='best', bbox_to_anchor=(0.5,-0.1))




























df_embeddings.plot.scatter(x = 'x', y = 'y')