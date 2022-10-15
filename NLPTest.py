#Install
#pip install spacy_sentence_bert
#pip install git+https://github.com/MartinoMensio/spacy-sentence-bert.git
#pip install spacy
#pip uninstall wasabi -y

#pip install -U sentence-transformers --user

#pip install numpy
#pip install pandas
#pip install wasabi==0.9.1

#pip install https://github.com/MartinoMensio/spacy-sentence-bert/releases/download/v0.1.2/en_stsb_distilbert_base-0.1.2.tar.gz#en_stsb_distilbert_base-0.1.2 --user

#pip install https://github.com/MartinoMensio/spacy-sentence-bert/releases/download/v0.1.2/en_stsb_bert_large-0.1.2.tar.gz#en_stsb_bert_large-0.1.2 --user

#pip install https://github.com/MartinoMensio/spacy-sentence-bert/releases/download/v0.1.2/en_stsb_roberta_large-0.1.2.tar.gz#en_stsb_roberta_large-0.1.2 --user

#pip install numpy --user
#pip install textblob
#pip install autocorrect
#pip install wordninja
#pip install textstat 

# Import 
import os
import numpy as np
import seaborn as sns
import pandas as pd
import spacy_sentence_bert
import matplotlib.pyplot as plt
import textblob
from autocorrect import Speller 
from textblob import TextBlob
import wordninja
import textstat 

# Get the current working directory

cwd = os.getcwd()
print(cwd)

# read by default 1st sheet of an excel file
df = pd.read_excel('SOPs v1.xlsx')

#from nltk.corpus import stopwords
#stop = stopwords.words('english')
#df["Text_wo"] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df["Text_wo"]= df["Text_wo"].str.replace(r'\W'," ")


print(df)

print(df.label.unique())

# OUTPUT
#['Procedure Name' 'Action (What)' 'Action (Where)' 'Action (How)' 'Decide (What)' 'Decide (How)' 'Waiting (What)' 'Waiting (How)' 'Trigger (What)' 'Actor' 'Trigger (How)' 'Trigger (Where)']

print(df.isnull().sum())
print(df2.isnull().sum())
df = df.dropna()

#Replacing

replacers = {"l/g":"landing gear", "hsc-manual":"high speed counter manual", "vnav":"vertical navigation", "lnav":"lateral navigation", "econ":"optimum descent speed", "flx":"reduced takeoff thrust", "mct":"maximum continuous thrust", "mcp":"maximum continuous power", "n1":"cockpit gauge which presents the rotational speed of the low pressure", "to/ga":"take-off go Around", "v/s":"stalling speed", "g/s":"ground Stop", "spd ":"speed mode", "flch":"flight level change", "alt":"altitude", "pth":"path", "atc":"Air traffic control", "ovrd ctr":"overdrive control traffic zone", "fl 180":"flight level", "navaids":"navigational Aids", "mcdu":"multi-function control and display unit", "fma":"flight mode annunciator", "hyd":"hydraulic", "rmps":"risk management process", "hdg":"heading the direction", "loc":"loss of aircraft control", "thr ref":"thrust reference", "cmd":"Command", "v1":"maximum speed at which a rejected takeoff can be done", "cdu":"control display units", "egt ":"exhaust gases temperature", "conf ":"configuration", "apu":"auxiliary power unit", "aft":"towards the rear", "pnf":"pilot not flying", " pf ":"pilot flying", "c":"captain", "pfd":"primary flight display", "f/o":"first officer", "egt":"temperature of the exhaust gases", "pu":"processing unit"}


df['text2'] = (df.text.str.replace('[...…]','')
    .str.split()
    .apply(lambda x: ' '.join([replacers.get(e, e) for e in x])))
        
   
        
        
#Sentiment 
df['sentiment'] = df['text2'].apply(lambda x: TextBlob(x).sentiment)

#Spell corrector

df['text2'] = df['text2'].apply(lambda x:str(TextBlob(x).correct()))

df['tag'] = df['text2'].apply(lambda x: TextBlob(x).tags)

#Remove short strings in text

df = df[df['text2'].str.len()>3]

#Replace Special Characters

df['text2'] = df['text2'].replace(r'[^\w\s]|_', '', regex=True)

#Remove extra spaces

df['text2'] = df['text2'].replace(r'\s+', ' ', regex=True)

#Word separation

df['text2'] = df['text2'].apply(lambda x: wordninja.split(x))

df['text2'] = df['text2'].apply(lambda x: " ".join(x))

#text stats

df['Readability_Index'] = df['text2'].apply(lambda x: textstat.automated_readability_index(x))
df['Reading_Time'] = df['text2'].apply(lambda x: textstat.reading_time(x))

# Frecuencies/Lengths

counts = df['label2'].value_counts()
counts.plot(kind='bar', legend=False, grid=True, figsize=(8, 5))

counts = df['label'].value_counts()
counts.plot(kind='bar', legend=False, grid=True, figsize=(8, 5))

lens = df.text.str.len()
lens.hist(bins = np.arange(0,200,5))


#BERT Word Vectors

#nlp = spacy_sentence_bert.load_model('en_stsb_bert_large')
#nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')
nlp = spacy_sentence_bert.load_model('en_stsb_roberta_large')

df['vector'] = df['text2'].apply(lambda x: nlp(x).vector)


#PCA Vector Dimensionality reduction
from sklearn.manifold import TSNE

X = list(df["vector"])

#X_embedded = TSNE(n_components=2).fit_transform(X) #2 dimension vector
X_embedded = TSNE(n_components=3).fit_transform(X) #3 dimension vector

df_embeddings = pd.DataFrame(X_embedded)
#df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})  #2 dimension vector
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y',2:'z'}) #3 dimension vector

df2 = pd.concat([df, df_embeddings], axis=1).reindex(df.index)

#Save clean file

df2.to_excel (r'C:\Users\EstebanEchandi\Desktop\SOPsv1Clean.xlsx', index = False, header=True)

#Check

len(df2.index)
df2.shape[0]
df2[df2.columns[0]].count()


#**************Dataset split***************
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(), df['label'].tolist(), test_size=0.3, random_state=690)



#**************LABEL***************
# train your choice of machine learning classifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)


#Test Algorithm

Examples = ['the takeoff must becontinued', 'if the door' ,'flight crew']

label = ['Action (What)', 'Decide (What)' ,'Actor']

for Examples, label in zip(Examples, label):
  print(Examples,)
  print(f"True Label: {label}, Predicted Label: {clf.predict(nlp(Examples).vector.reshape(1, -1))[0]} \n")



#**************LABEL***************
# train your choice of machine learning classifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=9, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)









#**************Dataset split***************

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['vector'].tolist(), df['label2'].tolist(), test_size=0.3, random_state=690)

#**************LABEL2***************
# train your choice of machine learning classifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

clf_report = classification_report(y_test,y_pred,output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, cmap="PiYG",annot=True)

#**************LABEL2***************
# train your choice of machine learning classifier

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

#X_embedded = TSNE(n_components=2).fit_transform(X) #2 dimension vector
X_embedded = TSNE(n_components=3).fit_transform(X) #3 dimension vector

df_embeddings = pd.DataFrame(X_embedded)
#df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})  #2 dimension vector
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y',2:'z'}) #3 dimension vector

df2 = pd.concat([df, df_embeddings], axis=1).reindex(df.index)

df2.to_excel (r'C:\Users\EstebanEchandi\Desktop\SOPsv1Clean.xlsx', index = False, header=True)

groups = df2.groupby("label")

for name, group in groups:
     #plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name,markersize=3)
     plt.plot(group["x"], group["y"], group["z"], marker="o", linestyle="", label=name,markersize=3)
plt.legend(loc='best', bbox_to_anchor=(0.5,-0.1))


groups2 = df2.groupby("label2")

for name, group in groups2:
    #plt.plot(group["x"], group["y"], marker="o", linestyle="", label=name,markersize=3)
    plt.plot(group["x"], group["y"], group["z"], marker="o", linestyle="", label=name,markersize=3)
plt.legend(loc='best', bbox_to_anchor=(0.5,-0.1))

# 3D Visualizations

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = Axes3D(fig)

y = group['y']
x = group['x']
z = group['z']
c = group['label2']

ax.scatter3D(x,y,z, cmap='coolwarm')
# setting title and labels
ax.set_title("3D plot")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)

y = group['y']
x = group['x']
z = group['z']
c = group['label']

ax.scatter3D(x,y,z, cmap='coolwarm')
# setting title and labels
ax.set_title("3D plot")
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()

