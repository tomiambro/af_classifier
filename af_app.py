# Basic
# import scipy
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster import hierarchy as hc
from scipy.stats import spearmanr
# MlFlow
import mlflow
import mlflow.sklearn
# Ploting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Metrics
from sklearn.metrics import f1_score, fbeta_score, make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


st.title('Atrial Fibrilation Detector')

"""
	Using the dataset provided by the 2020 Physionet Challenge we've developed an Atrial Fibrilation Detector trained to
	identify AF diagnosed patiences from a dataset containing patiances with different pathologies like: PAC, RBBB, I-AVB,
	PVC, LBBB, STD, STE and healthy individuals.

	Although data from 12-lead ECG was provided, for this first analysis we've only used the lead 2 data and we've processed
	the signals in order to create a dataframe consisting of features we believe will help us classify.
"""

@st.cache
def load_data(lead='lead2-HRV'):
	df = pd.read_feather(f'datasets/phys-raw-{lead}-corrected')
	df = df.loc[df['age'] >= 0]
	df.loc[df.label != 'AF', 'label'] = 'Non-AF'
	return df

@st.cache
def filter_df(df_raw):
	df = df_raw.copy()
	cols = df_raw.columns
	cols = cols.drop('label')
	for col in cols:
	    df = df[df[col] < df[col].quantile(.99)]
	df_raw = df.copy()
	return df_raw

df_raw = load_data()

st.header("Dendrogram")
corr = np.round(spearmanr(df_raw.drop('label', axis=1)).correlation, 4)
corr_condensed = hc.distance.squareform(1 - corr)
z = hc.linkage(corr_condensed, method='average')
fig_den = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_raw.drop('label',axis=1).columns, orientation='left', leaf_font_size=16)
st.pyplot(fig_den, clear_figure=True)

"""
We drop HRV_SDSD
"""
to_drop = ['HRV_SDSD']
df_raw = df_raw.drop(to_drop, axis=1)

st.header("Correlation Matrix")
fig_cor = plt.figure(figsize=(16,10))
sns.heatmap(df_raw.corr())
st.pyplot(fig_cor, clear_figure=True)

st.header("Boxplots")
fig_box1 = plt.figure(figsize=(20,5))
sns.boxplot(data=df_raw)
st.pyplot(fig_box1, clear_figure=True)

"""
### Let's remove some outliers
(We keep everything below the 99th percentile)
"""

#slider = st.sidebar.slider("", 800, 10100, 10100, 100)
df_raw = filter_df(df_raw)

fig_box2 = plt.figure(figsize=(20,5))
sns.boxplot(data=df_raw)
st.pyplot(fig_box2, clear_figure=True)

st.header("Pairplots")
# fig_pair = plt.figure(figsize=(20,17))
sns.pairplot(data=df_raw.sample(frac=0.1, random_state=42), hue='label', palette='Set2', height=1.5)
st.pyplot(clear_figure=True)

y = df_raw['label']
X = df_raw.drop('label', axis=1)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
res = pd.DataFrame({'model':[], 'f1':[]})

models = [LogisticRegression(), RandomForestClassifier(), SVC(), KNeighborsClassifier()]

for model in models:
	model.fit(X_train, y_train)
	f1 = f1_score(y_eval, model.predict(X_eval), pos_label='AF')
	res = res.append({'model': f"{model}", 'f1': f1}, ignore_index=True)

st.write(res.sort_values('f1', ascending=False))
print(res.sort_values('f1', ascending=False))