import sys
import numpy as np
import pandas as pd
from urllib.parse import urlparse
# MlFlow
import mlflow
import mlflow.sklearn
# Metrics
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import f1_score, fbeta_score, make_scorer, confusion_matrix
# Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_data(lead):
	df = pd.read_feather(f'datasets/phys-raw-{lead}-corrected')
	df = df.loc[df['age'] >= 0]
	df.loc[df.label != 'AF', 'label'] = 'Non-AF'
	return df

def filter_df(df, q):
	df = df.copy()
	cols = df.columns
	cols = cols.drop('label')
	for col in cols:
	    df = df[df[col] < df[col].quantile(q)]
	df = df.copy()
	return df

if __name__ == '__main__':
	lead = sys.argv[1] if len(sys.argv) > 1 else 'lead2-HRV'
	q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.99
	
	df_raw = load_data(lead)
	df_raw = filter_df(df_raw, q)