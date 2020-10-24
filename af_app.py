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

df_raw = load_data()

st.header("Dendrogram")
corr = np.round(spearmanr(df_raw.drop('label', axis=1)).correlation, 4)
corr_condensed = hc.distance.squareform(1 - corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_raw.drop('label',axis=1).columns, orientation='left', leaf_font_size=16)
st.pyplot(fig)

"""
We drop HRV_SDSD
"""
to_drop = ['HRV_SDSD']
df_raw = df_raw.drop(to_drop, axis=1)

st.header("Correlation")
fig = plt.figure(figsize=(16,10))
sns.heatmap(df_raw.corr())
st.pyplot(fig)

st.header("Boxplots")
df_raw = df_raw.drop(5685)
fig = plt.figure(figsize=(20,5))
sns.boxplot(data=df_raw)
st.pyplot(fig)


"""
Let's remove some outliers
"""
slider = st.slider("", 800, 10100, 10100, 1000)
df_raw.loc[df_raw.mean_P_Peaks > slider,'mean_P_Peaks']
df_raw = df_raw.drop(df_raw.loc[df_raw.mean_P_Peaks > slider,'mean_P_Peaks'].index)

fig = plt.figure(figsize=(20,5))
sns.boxplot(data=df_raw['mean_P_Peaks'])
st.pyplot(fig)