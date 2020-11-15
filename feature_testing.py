import sys, getopt
import subprocess
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
# ----------------------
bashCommand = 'rm -rf mlruns/.trash/*'
clear_flag = False
scaler_flag = False
lead = 'lead2-HRV'
q = 0.99

def load_data(lead):
	df = pd.read_feather(f'datasets/phys-raw-{lead}-eda')
	df.loc[df.label != 'AF', 'label'] = 'Non-AF'
	return df

def filter_df(df, q):
	df = df.copy()
	cols = df.columns
	cols = cols.drop('label')
	for col in cols:
	    df = df[df[col] < df[col].quantile(q)]
	return df

def no_op(train, val):
	return train, val

def scal_features(train, val):
	scal = StandardScaler()
	X_train = scal.fit_transform(train)
	X_eval = scal.transform(val)
	return X_train, X_eval

def pca_features(train, val):
	n_comps = train.shape[1]
	pca = PCA(n_components = n_comps)
	train, val = scal_features(train, val)
	X_train = pca.fit_transform(train)
	X_eval = pca.transform(val)
	return X_train, X_eval
# -----------------------


if __name__ == '__main__':

	arguments, values = getopt.getopt(sys.argv[1:],"cl:q:s",["clear", "lead", 'scal'])
	for current_argument, current_value in arguments:
		clear_flag = True if current_argument in ("-c", "--clear") else False
		scaler_flag = True if current_argument in ("-s", "--scal") else False
		lead = current_value if current_argument in ("-l", "--lead") else 'lead2-HRV'
		q = float(current_value) if current_argument in ("-q") else 0.99
	
	exp = mlflow.get_experiment_by_name('model_selection')

	if clear_flag:
		if exp != None and exp.lifecycle_stage != 'deleted':
			print('Previous experiment exists')
			mlflow.delete_experiment(exp.experiment_id)

		print(f'Archiving experiment with id {exp.experiment_id}')
		subprocess.run([f'ls -la mlruns/.trash/{exp.experiment_id}'], shell=True, check=True)
		
		try:
			subprocess.run(bashCommand, shell=True, check=True)
		except subprocess.CalledProcessError as e:
			print(e)
			exit(-1)
		experiment_id = mlflow.create_experiment('model_selection')
	else:
		experiment_id = mlflow.set_experiment('model_selection')

	df_raw = load_data(lead)
	df_raw = filter_df(df_raw, q)
	y = df_raw['label']
	X = df_raw.drop('label', axis=1)
	X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
	
	models = {
		'LR': LogisticRegression(max_iter = 2000, n_jobs = 4),
		'RF': RandomForestClassifier(n_jobs = 4),
		'SVC': SVC(),
		'KNN': KNeighborsClassifier(n_jobs = 4)
	}

	features = {
		'Original' : no_op,
		'StandardScaler' : scal_features,
		'PCA' : pca_features
	}
	
	# for function in features:

	for name, model in models.items():
		for fname, func in features.items():
			xtrain, xeval = func(X_train, X_eval)
			model.fit(xtrain, y_train)
			f1 = f1_score(y_eval, model.predict(xeval), pos_label='AF')

			with mlflow.start_run(experiment_id=experiment_id):
				print(f"Model {name}")
				print(f"  F1: {f1}")

				mlflow.log_param("model", f"{name}")
				mlflow.log_param("features", f"{fname}")
				mlflow.log_param("lead", lead)
				mlflow.log_param("quantile", q)
				mlflow.log_param("columns", list(X.columns))
				mlflow.log_param("number columns", len(list(X.columns)))
				mlflow.log_metric("f1", f1)

				tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

		        # Model registry does not work with file store
				if tracking_url_type_store != "file":

		            # Register the model
		            # There are other ways to use the Model Registry, which depends on the use case,
		            # please refer to the doc for more information:
		            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
					mlflow.sklearn.log_model(model, "model", registered_model_name="AF_Classifier")
				else:
					mlflow.sklearn.log_model(model, "model")