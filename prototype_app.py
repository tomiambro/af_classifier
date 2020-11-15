import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from scipy.cluster import hierarchy as hc
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, r2_score

def selectbox_without_default(label, options):
    options = [''] + options
    format_func = lambda x: 'Select one option' if x == '' else x
    return st.selectbox(label, options, format_func=format_func)

@st.cache
def load_data(lead='lead2-HRV'):
    df = pd.read_feather(f'datasets/phys-raw-{lead}-corrected')
    df = df.loc[df['age'] >= 0]
    df.loc[df.label != 'AF', 'label'] = 'Non-AF'
    return df

@st.cache
def filter_df(df_raw, q):
    df = df_raw.copy()
    cols = df_raw.columns
    cols = cols.drop('label')
    for col in cols:
        df = df[df[col] < df[col].quantile(q)]
    df_raw = df.copy()
    return df_raw

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

def main():
    # Title
    st.title("Model Experimentation with MLflow")

    # Choose dataset
    df_raw = load_data()
    st.write(df_raw.head())

    st.header("Dendrogram")
    corr = np.round(spearmanr(df_raw.drop('label', axis=1)).correlation, 4)
    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr_condensed, method='average')
    fig_den = plt.figure(figsize=(16,10))
    dendrogram = hc.dendrogram(z, labels=df_raw.drop('label',axis=1).columns, orientation='left', leaf_font_size=16)
    st.pyplot(fig_den, clear_figure=True)

    st.header("Correlation Matrix")
    fig_cor = plt.figure(figsize=(16,10))
    sns.heatmap(df_raw.corr())
    st.pyplot(fig_cor, clear_figure=True)

    st.header("Boxplots")
    fig_box1 = plt.figure(figsize=(20,5))
    sns.boxplot(data=df_raw)
    st.pyplot(fig_box1, clear_figure=True)

    q = st.slider("", 0.9, 1.0, 0.99, 0.01)
    df_raw = filter_df(df_raw, q)

    fig_box2 = plt.figure(figsize=(20,5))
    sns.boxplot(data=df_raw)
    st.pyplot(fig_box2, clear_figure=True)

    # Model selection
    models = {
        'Logistic Regression': LogisticRegression(max_iter = 2000, n_jobs = 4, random_state=42),
        'Random Forest': RandomForestClassifier(n_jobs = 4, random_state=42),
        'SVC': SVC(random_state=42),
        'KNNeighbors': KNeighborsClassifier(n_jobs = 4)
    }

    # Feature selection
    feature_options = df_raw.columns.drop('label').tolist()
    feature_choice = st.multiselect("Choose features to drop", feature_options)
    treatment_options = {
        'None': no_op,
        'StandardScaler' : scal_features,
        'PCA' : pca_features
    }

    treatment_choice = st.selectbox("Choose feature treatment", list(treatment_options.keys()))
    clear_mlflow = st.checkbox("Clear mlflow experiments?")
    # Mlflow tracking
    track_with_mlflow = st.checkbox("Track with mlflow?")

    # Model training
    start_training = st.button("Start training")
    if not start_training:
        st.stop()


    y = df_raw['label'].copy()
    sub_df = df_raw.drop([*feature_choice, 'label'], axis=1)
    X = sub_df.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    res = pd.DataFrame({'model':[], 'f1':[]})

    for name, model in models.items():
        if track_with_mlflow:
            mlflow.set_experiment(data_choice)
            mlflow.start_run()
            mlflow.log_param('features', feature_choice)
            mlflow.log_param('model', name)

        X_train, X_test = treatment_options[treatment_choice](X_train, X_test)
        st.write(f'Training {name}')
        model.fit(X_train, y_train)

        # Model evaluation
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)
        metric_name = "f1_score"
        metric_train = f1_score(y_train, preds_train, pos_label='AF')
        metric_test = f1_score(y_test, preds_test, pos_label='AF')
        
        # st.write(f"{metric_name}_train", round(metric_train, 3))
        # st.write(f"{metric_name}_test", round(metric_test, 3))
        res = res.append({'model': f"{name}", 'f1': metric_test}, ignore_index=True)

        if track_with_mlflow:
            mlflow.log_metric(metric_name+"_train", metric_train)
            mlflow.log_metric(metric_name+"_test", metric_test)
            mlflow.end_run()

    st.write(res.sort_values('f1', ascending=False))

if __name__ == '__main__':
    main()