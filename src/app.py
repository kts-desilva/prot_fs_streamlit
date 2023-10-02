# %%
import pandas as pd
import streamlit as st
import eda
import numpy as np
import os
import tensorflow as tf
# os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from feature_models import create_model, FeatureCreation
import pickle
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import load_model
import functools
from sklearn.model_selection import train_test_split
import graphs

#Basic libraries
from scipy import stats

#import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn2

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, cross_val_predict,  KFold
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix,RocCurveDisplay

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from sklearn.metrics import accuracy_score

from sklearn.metrics import RocCurveDisplay

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from yellowbrick.model_selection import RFECV
from streamlit_yellowbrick import st_yellowbrick
from sklearn.feature_selection import SequentialFeatureSelector

# %%
st.set_page_config(
    page_title="Selecting New Unique Protoemic Markers",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

#st.set_option('deprecation.showPyplotGlobalUse', False)

# ----------- Data -------------------


#@st.cache
def get_raw_data():
    """
    This function returns a pandas DataFrame with the raw data.
    """

    #raw_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'ovarian_clinical_data2.csv'))
    
    uploaded_file = st.sidebar.file_uploader("Upload Dataset",type=["csv","xlsx","xls"])
    raw_df = None 

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        #st.write(raw_data)
    else:
        raw_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'zilulu_filtered_data.csv'))
    
    return raw_df


#@st.cache
def get_cleaned_data():
    """
    This function return a pandas DataFrame with the cleaned data.
    """

    clean_data = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'houses_to_rent_v2_fteng.csv'))
    return clean_data

#@st.cache
def get_processed_data():
    """
    This function return a pandas DataFrame with the processed data.
    """

    #processed_data = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'processed_data.csv'))
    processed_data = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'processed_data2.csv'))
    return processed_data

def get_annotation_data():
    """
    This function return a pandas DataFrame with the annotation data.
    """

    #annotation_data = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'annotation_data.csv'))
    annotation_data = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'data_annotation2.csv'))
    return annotation_data

#@st.cache
def get_raw_eval_df():
    """
    This function return a pandas DataFrame with the dataframe and the machine learning models along with it's metrics.
    """

    raw_eval_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'model_evaluation.csv'))
    return raw_eval_df


#@st.cache(hash_funcs={pd.DataFrame: lambda x: x})
# def load_models_df(dataframe):
#     df_evaluated = dataframe.copy()
#     models_list = os.listdir(os.path.join(os.path.abspath(''), 'models'))
#     rep = {"pipe": "model", "pickle": "h5"}
#     for index, row in df_evaluated.iterrows():
#         # check if the file_name is in our models directory
#         if row['pipe_file_name'] in models_list:
#             # now, load the model.
#             with open(os.path.join(os.path.abspath(''), 'models', row['pipe_file_name']), 'rb') as fid:
#                 model_trained = pickle.load(fid)
            
#             # for the keras model, we have to load the model separately and add into the pipeline or transformed target object.
#             if row['name'] == 'NeuralNetwork':
#                 model_keras = load_model(os.path.join(os.path.abspath(''), 'models', functools.reduce(lambda a, kv: a.replace(*kv), rep.items(), row['pipe_file_name'])))
#                 # check if the target transformer it is active
#                 if row['custom_target']:
#                     # reconstruct the model inside a kerasregressor and add inside the transformed target object
#                     model_trained.regressor.set_params(model = KerasRegressor(build_fn=create_model, verbose=0))
#                     # add the keras model inside the pipeline object
#                     model_trained.regressor_.named_steps['model'].model = model_keras
#                 else:
#                     model_trained.named_steps['model'].model = model_keras

#             df_evaluated.loc[index, 'model_trained'] = model_trained

#     # we have to transform our score column to bring it back to a python list
#     df_evaluated['all_scores_cv'] = df_evaluated['all_scores_cv'].apply(lambda x: [float(i) for i in x.strip('[]').split()])
    
#     return df_evaluated.sort_values(by='rmse_cv').reset_index(drop=True)


#@st.cache
def split(dataframe):
    df = dataframe.copy()
    x = df.drop(columns=['rent amount (R$)'], axis=1)
    y = df['rent amount (R$)']
    # check if the random state it is equal to when it was trained, this is very important.
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=0)

    return x, y, x_train, x_test, y_train, y_test


clean_df = get_cleaned_data()
raw_eval_df = get_raw_eval_df()
#eval_df = load_models_df(raw_eval_df)
x, y, x_train, x_test, y_train, y_test = split(clean_df)
processed_data = get_processed_data()
annotation_data =  get_annotation_data()

def preprocess_data(df, id_column, char_col,class_of_interest,control_class):
    #df["Binary_Class"] = np.select([df["Sample_Tumor_Normal"] == "Tumor",df["Sample_Tumor_Normal"] == "Normal"],[ 1, 0])
    #df = df[not df[char_col] in ["B_V1","A_V1"]]
    df = df[df[char_col].isin([class_of_interest,control_class])]
    
    df["Binary_Class"] = np.select([df[char_col] == control_class,df[char_col] == class_of_interest],[ 0, 1])
    df.fillna(0, inplace=True)
    
    #unwanted_columns = ['Patient_ID','Sample_Tumor_Normal','Binary_Class' ]
    unwanted_columns = [id_column,char_col,'Binary_Class' ]

    # data splitting
    X_combin = df.drop(unwanted_columns, axis=1)
    y = df[['Binary_Class']]

    X_combin = X_combin.loc[:,~X_combin.columns.duplicated()]
    
    return (X_combin,y) 

# ----------- Global Sidebar ---------------

st.sidebar.title("UMAI: Select New Unique Marker AI")

condition = st.sidebar.selectbox(
    "Select the visualization",
    ("Introduction", "EDA", "Feature Selection", "Model Prediction")
)
    
# ------------- Introduction ------------------------

if condition == 'Introduction':
    st.image(os.path.join(os.path.abspath(''), 'data', 'histology.jpg'))
    st.subheader('UMAI Introduction')
    
    ## FALTA O CHECK ON GITHUB
    st.write("""
    This application provides an overview of the ovarian cancer proteomics data from CPTAC data portal. It is a dataset that provides protein expression profiles of ovarian tumor and control samples.

    The app it is [deployed](https://kts-desilva-prot-fs-streamlit-srcapp-eh8dfx.streamlit.app/) in Streamlit.

    The data were provided from this [source](https://cptac-data-portal.georgetown.edu/). 

    You can check on the sidebar of the app:
    - EDA (Exploratory Data Analysis)
    - Feature Selection
    - Model Prediction
    - Model Evaluation

    The predictions are made with classification performed to distinguish tumor vs control utlizing pre-trained machine learning models.

    The data is available in raw format and pre-processed format as csv files inside the data directory. If you want to check the code, go through the notebook directory in github repository. [github repository](https://github.com/kts-desilva/prot_fs_streamlit).
    """)
    
    st.subheader('Feature Selection')

    st.write("""
    Feature selection based on recursive and iterative feature selection methods based on scikit-learn packages.
    Machine learning algorithms used

    - XGBoost
    - Random Forest
    - Support Vector Machine
    - Stochastic Gradient Descent
    """)

    st.subheader('Model Definition')

    st.write("""
    The structure of the training it is to wrap the process around a scikit-learn Pipeline. There were 4 possible combinations and 5 models, resulting in 20 trained models.

    The combinations are regarding to perform Feature Creation and/or Target Transformations in the dataset.

    Models:

    - XGBoost
    - Random Forest
    - Support Vector Machine
    - Stochastic Gradient Descent
    
    Our main accuracy metric is RMSE. To enhance our model definition, we utilized cross-validation and Random Search for hyperparameter tuning.
    Further, we have considered using precision, recall and specificity metrics to assess the quality of the developed methodology.
    """)
    
    st.image(os.path.join(os.path.abspath(''), 'data', 'ovarian-Methodology.png'))

# ------------- EDA ------------------------

elif condition == 'EDA':
    data = get_raw_data()
    with st.container():
        st.header('Descriptive Statistics\n')
        col1, col2 = st.columns([1, 3])
        col1.dataframe(eda.summary_table(data))
        col2.dataframe(data.describe())

    st.header('Data Visualization')

    height, width, margin = 450, 1500, 10

    st.subheader('Disease Proteomics Distribution')

    select_city_eda = st.selectbox(
        'Select the Disease Type',
        [i for i in data['Sample_Tumor_Normal'].unique()]
    )
    
    ncolms =   list(data.columns.values.tolist())
    ncolms.remove('Sample_Tumor_Normal')
    ncolms.remove('Patient_ID')
    
    select_protein_eda = st.selectbox(
        'Select Protein',
        [i for i in ncolms]
    )

    if select_city_eda == 'All':
        fig = graphs.plot_histogram(data=data, x=select_protein_eda, nbins=50, height=height, width=width, margin=margin)
    else:
        fig = graphs.plot_histogram(
            data = data.loc[data['Sample_Tumor_Normal'] == select_city_eda], x=select_protein_eda, nbins=50, height=height, width=width, margin=margin)
                      
    st.plotly_chart(fig)

    st.subheader('Histogram for Protein')

    fig = graphs.plot_boxplot(data=data, x="Sample_Tumor_Normal", y=select_protein_eda, color="Sample_Tumor_Normal", height=height, width=width, margin=margin)

    st.plotly_chart(fig)
    
# -------------------------------------------

elif condition == 'Feature Selection':
    raw_df = get_raw_data()
    character_columns = raw_df.select_dtypes(include=['object']).columns

    # Create a select box for character columns
    id_column = st.sidebar.selectbox('Select the id variable column:', character_columns)
    class_column = st.sidebar.selectbox('Select the class variable column:', character_columns)
    class_of_interest = st.sidebar.selectbox('Select class of interest:', raw_df[class_column].unique().tolist())
    control_class = st.sidebar.selectbox('Select control class:', raw_df[class_column].unique().tolist())

    X_combin,y = preprocess_data(raw_df, id_column, class_column,class_of_interest,control_class)
    X_train, X_test, y_train, y_test = train_test_split(X_combin, y, test_size=0.33, random_state=0)
    height, width, margin = 450, 1500, 25
    
#     st.subheader('Intial Inspection with XGBoost Classifier')
    
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     model3 = RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100)
#     scores = cross_val_score(model3, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
#     model3.fit(X_train, y_train)
    
#     fig, ax = plt.subplots(figsize=(3.5, 1.5))
#     RocCurveDisplay.from_estimator(model3, X_test, y_test)
#     st.pyplot(fig)
    
#     y_pred_proba = model3.predict_proba(X_test)[::,1]
#     fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
#     fig =  graphs.plot_roc(fpr, tpr,height, width, margin)
#     st.plotly_chart(fig)
    
    st.sidebar.text('Select Feature Selection Methods')
    rfe = st.sidebar.checkbox('Recursive Feature Elimination', value=True)
    sfs = st.sidebar.checkbox('Sequential Feature Selection', value=True)
    
    ecv = st.sidebar.checkbox('Enable cross validation', value=True)
    
    if(rfe):
        st.subheader('Recursive Feature Elimination (RFE)')
        options = st.sidebar.multiselect(
        'Select RFE Algorithms',
        ['Random Forest', 'Support Vector Machine', 'Stochastic Gradient Descent Classifier', 'XGBoost'],
        default  = ['Random Forest', 'Support Vector Machine', 'Stochastic Gradient Descent Classifier'])

        st.subheader('Recursive Feature Elimination with SGDClassifier')
        # SGDClassifier
        prots = []
        with open(os.path.join(os.path.abspath(''), 'data', 'diff_exp_proteins.txt')) as f:
            prots = f.read().splitlines()
        #new_df = X_combin[prots]
        new_df = X_combin
    #     new_df = X_combin[['SYNM','LAMB2','ITGA7','TNS1',
    #                    'HSPB6','DMD','OGN','PGM5','CAVIN2','SOD3',
    #                    'SORBS1','NID1','SORBS1','ABCA8','TNS2','CD34']]
        visualizer = RFECV(SGDClassifier(max_iter=1000, tol=1e-3))
        visualizer.fit(new_df, y)        # Fit the data to the visualizer
        #visualizer.show()
        st_yellowbrick(visualizer)  
        new_df2_sdg = new_df.loc[:, visualizer.support_]
        st.text("SGDClassifier Features: ")
        #st.text(new_df2_sdg.columns)
        st.text(', '.join(new_df2_sdg.columns))

        #rf-taking too much time
        st.subheader('Recursive Feature Elimination with Random Forest')
        cv_rf = StratifiedKFold(3)
        visualizer_rf = RFECV(RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100), cv=cv_rf, scoring='f1_weighted')
        visualizer_rf.fit(new_df, y)
        #visualizer_rf.show()
        st_yellowbrick(visualizer_rf) 
        new_df2_rf = new_df.loc[:, visualizer_rf.support_]
        print("Features: ", new_df2_rf.columns)
        #find_if_correct_features_found(new_df2.columns)
        st.text("Random Forest Features: ")
        #st.text(new_df2_rf.columns)
        st.text(', '.join(new_df2_rf.columns))

        #svm
        st.subheader('Recursive Feature Elimination with Support Vector Machine')
        visualizer = RFECV(SVC(kernel='linear', C=1))
        visualizer.fit(new_df, y)
        #visualizer.show()
        st_yellowbrick(visualizer) 
        new_df2_svm = new_df.loc[:, visualizer.support_]
        print("Features: ", new_df2_svm.columns)
        #find_if_correct_features_found(new_df2.columns)
        st.text("Support Vector Machine Features: ")
        #st.text(new_df2_svm.columns)
        st.text(', '.join(new_df2_svm.columns))

        #xgb
    #     xgb1 = XGBClassifier(
    #         learning_rate =0.2,
    #         n_estimators=1000,
    #         max_depth=5,
    #         min_child_weight=1,
    #         gamma=0,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         objective= 'binary:logistic',
    #         nthread=4,
    #         scale_pos_weight=1,
    #         seed=27)
    #     visualizer = RFECV(xgb1)
    #     visualizer.fit(new_df, y)
    #     #visualizer.show() 
    #     st_yellowbrick(visualizer) 
    #     new_df2_xgb = new_df.loc[:, visualizer.support_]
    #     print("Features: ", new_df2.columns)
    #     st.text("XGBoost Features: "+ new_df2_xgb.columns)

        set1 = set(new_df2_sdg.columns)
        set2 = set(new_df2_svm.columns)
        set3 = set(new_df2_rf.columns)

        print(set1)
        print(set2)
        print(set3)
        fig, ax = plt.subplots(figsize=(5, 5))
        venn3([set1, set2, set3], ('SGD', 'SVM', 'RF'))
        st.pyplot(fig)
    
    if(sfs):
        options = st.sidebar.multiselect(
        'Select SFS Algorithms',
        ['Random Forest', 'Support Vector Machine', 'Stochastic Gradient Descent Classifier','XGBoost'],
        default=['Stochastic Gradient Descent Classifier','XGBoost'])

        st.subheader('Sequential Feature Selector: SGDClassifier')
        #Selecting the Best important features according to Logistic Regression
        # new_df3 = X_combin[['SYNM','LAMB2','ITGA7','TNS1','OGN','PGM5','CAVIN2','SOD3',
        #                'SORBS1','NID1']] 
        new_df3 = X_combin[['K7ES00_H3.3B', 'U3KQK0_H2BC15', 'A0A0C4DH24_IGKV6.21', 'P01814_IGHV2.70', 'A0A0B4J1X5_IGHV3.74']] 
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        sfs_selector = SequentialFeatureSelector(estimator=SGDClassifier(max_iter=1000, tol=1e-3), n_features_to_select = 4, cv =cv, direction ='forward')
        sfs_selector.fit(new_df3, y)
        new_df4_sgd = new_df3.loc[:, sfs_selector.support_]
        st.text("SGD Features: ")
        st.text(', '.join(new_df4_sgd.columns))

        xgb1 = XGBClassifier(
            learning_rate =0.2,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective= 'binary:logistic',
            nthread=4,
            scale_pos_weight=1,
            seed=27)

        sfs_selector = SequentialFeatureSelector(estimator=xgb1, n_features_to_select = 4, cv =cv, direction ='forward')
        sfs_selector.fit(new_df3, y)
        new_df4_xgb = new_df3.loc[:, sfs_selector.support_]
        st.text("XGB Features: ")
        #st.text(new_df4_xgb.columns)
        st.text(', '.join(new_df4_xgb.columns))

        set3 = set(new_df4_sgd.columns)
        set4 = set(new_df4_xgb.columns)

        print(set3)
        print(set4)
        fig, ax = plt.subplots(figsize=(5, 5))
        venn2([set3, set4], ('SGD', 'XGB'))
        st.pyplot(fig)
    
    #new_df4 = X_combin[['SYNM','LAMB2','OGN','SOD3']]
    new_df4 = X_combin[[set.intersection(set3,set4)]]

# -------------------------------------------

elif condition == 'Model Prediction':
    st.subheader('Classification Models')
    
    options = st.sidebar.multiselect(
        'Select Classification Algorithms',
        ['Random Forest', 'Support Vector Machine', 'Stochastic Gradient Descent Classifier', 'XGBoost'],
        default  = ['Random Forest', 'Support Vector Machine', 'Stochastic Gradient Descent Classifier'])
    
    height, width, margin = 450, 1500, 25
    
    #todo: add pre-trained model option also
    
    df = processed_data.copy()
    y = annotation_data.copy()
    st.write(df.head())
    st.write(y)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=0)

    xgb2 = XGBClassifier(
     learning_rate =0.2,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=1,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(xgb2, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    xgb2.fit(X_train, y_train)
    
    y_pred_proba = xgb2.predict_proba(X_test)[::,1]
    fpr, tpr, _ =  metrics.roc_curve(y_test,  y_pred_proba)
    fig =  graphs.plot_roc(fpr, tpr,height, width, margin)
    st.text("XGB ROC Curve: ")
    st.plotly_chart(fig)
    
    model3 = RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100)
    scores = cross_val_score(model3, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    model3.fit(X_train, y_train)
    y_pred_proba_rf = model3.predict_proba(X_test)[::,1]
    fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test,  y_pred_proba_rf)
    fig_rf =  graphs.plot_roc(fpr_rf, tpr_rf,height, width, margin)
    st.text("Random Forest ROC Curve: ")
    st.plotly_chart(fig_rf)
    
    model2 = SVC(kernel='linear', C=1,probability=True)
    scores = cross_val_score(model2, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    model2.fit(X_train, y_train)
    y_pred_proba_svm = model2.predict_proba(X_test)[::,1]
    fpr_svm, tpr_svm, _ = metrics.roc_curve(y_test,  y_pred_proba_svm)
    fig_svm =  graphs.plot_roc(fpr_svm, tpr_svm,height, width, margin)
    st.text("SVM ROC Curve: ")
    st.plotly_chart(fig_svm)

# -------------------------------------------

