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
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import load_model
import functools
from sklearn.model_selection import train_test_split
import graphs

#Basic libraries
from scipy import stats

#import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

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

# %%
st.set_page_config(
    page_title="Predicting Real Estate Prices in Brazil",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

#st.set_option('deprecation.showPyplotGlobalUse', False)

# ----------- Data -------------------


#@st.cache
def get_raw_data():
    """
    This function return a pandas DataFrame with the raw data.
    """

    raw_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'ovarian_clinical_data2.csv'))
    return raw_df


#@st.cache
def get_cleaned_data():
    """
    This function return a pandas DataFrame with the cleaned data.
    """

    clean_data = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'houses_to_rent_v2_fteng.csv'))
    return clean_data


#@st.cache
def get_raw_eval_df():
    """
    This function return a pandas DataFrame with the dataframe and the machine learning models along with it's metrics.
    """

    raw_eval_df = pd.read_csv(os.path.join(os.path.abspath(''), 'data', 'model_evaluation.csv'))
    return raw_eval_df


#@st.cache(hash_funcs={pd.DataFrame: lambda x: x})
def load_models_df(dataframe):
    df_evaluated = dataframe.copy()
    models_list = os.listdir(os.path.join(os.path.abspath(''), 'models'))
    rep = {"pipe": "model", "pickle": "h5"}
    for index, row in df_evaluated.iterrows():
        # check if the file_name is in our models directory
        if row['pipe_file_name'] in models_list:
            # now, load the model.
            with open(os.path.join(os.path.abspath(''), 'models', row['pipe_file_name']), 'rb') as fid:
                model_trained = pickle.load(fid)
            
            # for the keras model, we have to load the model separately and add into the pipeline or transformed target object.
            if row['name'] == 'NeuralNetwork':
                model_keras = load_model(os.path.join(os.path.abspath(''), 'models', functools.reduce(lambda a, kv: a.replace(*kv), rep.items(), row['pipe_file_name'])))
                # check if the target transformer it is active
                if row['custom_target']:
                    # reconstruct the model inside a kerasregressor and add inside the transformed target object
                    model_trained.regressor.set_params(model = KerasRegressor(build_fn=create_model, verbose=0))
                    # add the keras model inside the pipeline object
                    model_trained.regressor_.named_steps['model'].model = model_keras
                else:
                    model_trained.named_steps['model'].model = model_keras

            df_evaluated.loc[index, 'model_trained'] = model_trained

    # we have to transform our score column to bring it back to a python list
    df_evaluated['all_scores_cv'] = df_evaluated['all_scores_cv'].apply(lambda x: [float(i) for i in x.strip('[]').split()])
    
    return df_evaluated.sort_values(by='rmse_cv').reset_index(drop=True)


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

raw_df = get_raw_data()
clean_df = get_cleaned_data()
raw_eval_df = get_raw_eval_df()
#eval_df = load_models_df(raw_eval_df)
x, y, x_train, x_test, y_train, y_test = split(clean_df)

def preprocess_data(df):
    df.shape
    df["Binary_Class"] = np.select([df["Sample_Tumor_Normal"] == "Tumor",df["Sample_Tumor_Normal"] == "Normal"],[ 1, 0])
    df.fillna(0, inplace=True)
    
    unwanted_columns = ['Patient_ID','Sample_Tumor_Normal','Binary_Class' ]

    # data splitting
    X_combin = df.drop(unwanted_columns, axis=1)
    y = df[['Binary_Class']]

    X_combin = X_combin.loc[:,~X_combin.columns.duplicated()]
    
    return (X_combin,y) 

# ----------- Global Sidebar ---------------

condition = st.sidebar.selectbox(
    "Select the visualization",
    ("Introduction", "EDA", "Feature Selection", "Model Prediction", "Model Evaluation")
)

# ------------- Introduction ------------------------

if condition == 'Introduction':
    st.image(os.path.join(os.path.abspath(''), 'data', 'histology.jpg'))
    st.subheader('About')
    
    ## FALTA O CHECK ON GITHUB
    st.write("""
    This application provides an overview of the ovrain cancer proteomics data from CPTAC data portal. It is a dataset that provides protein expression profiles of ovarian tumor and control samples.

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
    
    Our main accuracy metric is RMSE. To enhance our model definition, we utilized Cross Validation and Random Search for hyperparameter tuning.
    Further, we have considered using precision, recall and specificity metrics to access the quality of the developed methodlogy.
    """)

# ------------- EDA ------------------------

elif condition == 'EDA':
    type_of_data = st.radio(
        "Type of Data",
        ('Raw Data', 'Cleaned Data'),
        help='Data source that will be displayed in the charts'
    )

    if type_of_data == 'Raw Data':
        data = raw_df.copy()
    else:
        data = clean_df.copy()

    with st.beta_container():
        st.header('Descriptive Statistics\n')
        col1, col2 = st.beta_columns([1, 3])
        col1.dataframe(eda.summary_table(data))
        col2.dataframe(data.describe())

    st.header('Data Visualization')

    height, width, margin = 450, 1500, 10

    st.subheader('Diasese Proteomics Distribution')

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
    X_combin,y = preprocess_data(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(X_combin, y, test_size=0.33, random_state=0)
    height, width, margin = 450, 1500, 25
    
    st.subheader('Intial Inspection with XGBoost Classifier')
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    model3 = RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100)
    scores = cross_val_score(model3, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    model3.fit(X_train, y_train)
    
#     fig, ax = plt.subplots(figsize=(3.5, 1.5))
#     RocCurveDisplay.from_estimator(model3, X_test, y_test)
#     st.pyplot(fig)
    
    y_pred_proba = model3.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    fig =  graphs.plot_roc(fpr, tpr,height, width, margin)
    st.plotly_chart(fig)
    
    
    st.subheader('Recursive Feature Elimination with SGDClassifier')
    # SGDClassifier
    prots = []
    with open(os.path.join(os.path.abspath(''), 'data', 'diff_exp_proteins.txt')) as f:
        prots = f.read().splitlines()
    new_df = X_combin[prots]
#     new_df = X_combin[['SYNM','LAMB2','ITGA7','TNS1',
#                    'HSPB6','DMD','OGN','PGM5','CAVIN2','SOD3',
#                    'SORBS1','NID1','SORBS1','ABCA8','TNS2','CD34']]
    visualizer = RFECV(SGDClassifier(max_iter=1000, tol=1e-3))
    visualizer.fit(new_df, y)        # Fit the data to the visualizer
    #visualizer.show()
    st_yellowbrick(visualizer)  
    new_df2_sdg = new_df.loc[:, visualizer.support_]
    st.text("SGDClassifier Features: "+ new_df2_sdg.columns)
    
    #rf-taking too much time
#     st.subheader('Recursive Feature Elimination with Random Forest')
#     cv_rf = StratifiedKFold(3)
#     visualizer_rf = RFECV(RandomForestClassifier(), cv=cv_rf, scoring='f1_weighted')
#     visualizer_rf.fit(new_df, y)
#     #visualizer_rf.show()
#     st_yellowbrick(visualizer_rf) 
#     new_df2 = new_df.loc[:, visualizer_rf.support_]
#     print("Features: ", new_df2.columns)
#     #find_if_correct_features_found(new_df2.columns)
#     st.text("Random Forest Features: "+ new_df2.columns)
    
    #svm
    st.subheader('Recursive Feature Elimination with Support Vector Machine')
    visualizer = RFECV(SVC(kernel='linear', C=1))
    visualizer.fit(new_df, y)
    #visualizer.show()
    st_yellowbrick(visualizer) 
    new_df2_svm = new_df.loc[:, visualizer.support_]
    print("Features: ", new_df2_svm.columns)
    #find_if_correct_features_found(new_df2.columns)
    st.text("Support Vector Machine Features: "+ new_df2_svm.columns)
    
    #xgb
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
    visualizer = RFECV(xgb1)
    visualizer.fit(new_df, y)
    #visualizer.show() 
    st_yellowbrick(visualizer) 
    new_df2_xgb = new_df.loc[:, visualizer.support_]
    print("Features: ", new_df2.columns)
    st.text("XGBoost Features: "+ new_df2_xgb.columns)
    
    set1 = set(new_df2_sdg.columns)
    set2 = set(new_df2_svm.columns)
    set3 = set(new_df2_xgb.columns)

    venn3([set1, set2, set3], ('SDG', 'SVM', 'XGB'))
    st.pyplot()
    

# -------------------------------------------

elif condition == 'Model Prediction':

    select_model_mpredict = st.sidebar.selectbox(
        'Select the Model',
        [i for i in eval_df['name'].unique()]  
    )

    select_custom_features_mpredict = st.sidebar.select_slider(
        'Create Custom Features?',
        [False, True],
        help='Feature Creation according to the FeatureCreation class in the load_models module'
    )

    select_custom_target_mpredict = st.sidebar.select_slider(
        'Perform Target Transformation?',
        [False, True],
        help='Perform a logarithm transformation in the target variable'
    )

    select_city = st.sidebar.selectbox(
        'Select the City',
        clean_df['city'].value_counts().index
    )

    select_area = st.sidebar.number_input(
        'Select the value of Area',
        help='The value must be in square meters (m²)',
        min_value=1,
    )

    select_rooms = st.sidebar.number_input(
        'Select the number of Rooms',
        min_value=1,
    )

    select_bathrooms = st.sidebar.number_input(
        'Select the number of Bathrooms',
        min_value=1,
    )

    select_parking_spaces = st.sidebar.number_input(
        'Select the number of Parking Spaces',
        min_value=0,
    )

    select_animal = st.sidebar.select_slider(
        'Accept Animals?',
        ['acept', 'not acept']
    )

    select_furniture = st.sidebar.select_slider(
        'It is furnished',
        ['furnished', 'not furnished']
    )

    select_hoa = st.sidebar.number_input(
        'Select the value of Hoa',
        help='The values must be in Reais (R$)',
        min_value=0,
    )

    select_property_tax = st.sidebar.number_input(
        'Select the value of Property Tax',
        help='The values must be in Reais (R$)',
        min_value=0,
    )

    select_fire_insurance = st.sidebar.number_input(
        'Select the value of Fire Insurance',
        help='The values must be in Reais (R$)',
        min_value=0,
    )

    predict_array = [select_city, select_area, select_rooms, select_bathrooms, select_parking_spaces, select_animal, select_furniture, select_hoa, select_property_tax, select_fire_insurance]

    model_trained_mpredict = eval_df.loc[(eval_df['name'] == select_model_mpredict) & (eval_df['custom_features'] == select_custom_features_mpredict) & (eval_df['custom_target'] == select_custom_target_mpredict)]['model_trained'].iloc[0]

    value_to_predict = pd.DataFrame(
        [predict_array], columns=clean_df.drop(columns='rent amount (R$)').columns
    )

    st.subheader('Available Models')

    st.dataframe(eval_df.drop(columns=['all_scores_cv', 'pipe_file_name', 'model_trained']))

    if st.button('Predict', help='Be certain to check the parameters on the sidebar'):
        predicted_value = model_trained_mpredict.predict(value_to_predict)
        st.success(f'The predicted value is R$ {round(predicted_value[0], 2)}')

        with st.beta_expander("Model Parameters"):
            st.write(f"The model chosen was {select_model_mpredict}. \n\n Parameters:", eval(eval_df.loc[(eval_df['name'] == select_model_mpredict) & (eval_df['custom_features'] == select_custom_features_mpredict) & (eval_df['custom_target'] == select_custom_target_mpredict)]['params'].iloc[0])[0])


# -------------------------------------------

elif condition == 'Model Evaluation':
    st.subheader('Available Models')

    st.dataframe(eval_df.drop(columns=['all_scores_cv', 'pipe_file_name', 'model_trained']))

    select_model_meval = st.sidebar.selectbox(
        'Select the Model',
        [i for i in eval_df['name'].unique()]  
    )

    select_custom_features_meval = st.sidebar.select_slider(
        'Create Custom Features?',
        [False, True]
    )

    select_custom_target_meval = st.sidebar.select_slider(
        'Perform Target Transformation?',
        [False, True]
    )

    model_trained_meval = eval_df.loc[(eval_df['name'] == select_model_meval) & (eval_df['custom_features'] == select_custom_features_meval) & (eval_df['custom_target'] == select_custom_target_meval)]['model_trained'].iloc[0]

# -------------- figs -----------------

    height, width, margin = 450, 1500, 30

    st.subheader('Distribution of the Target Variable')

    fig = graphs.plot_distplot(
        y_real=y_test, 
        y_predict=model_trained_meval.predict(x_test),
        height=height, 
        width=width, 
        margin=margin,
        title_text='Predicted and Real Value'
    )

    st.plotly_chart(fig)

    st.subheader('Distribution of the Residuals')

    # predict the values of the entire data
    prediction = model_trained_meval.predict(x)
    # calculate the residual
    resid = prediction - y

    # create a copy to not alter the original data
    df_plot = clean_df.copy()
    # create a column to identify the data regarding to train or test
    df_plot['split'] = 'train'
    df_plot.loc[x_test.index, 'split'] = 'test'
    df_plot['prediction'] = prediction
    df_plot['resid'] = resid

    # plot the residual plot with the histograms
    fig = graphs.plot_scatter(data=df_plot, x='prediction', y='resid', residual=True, height=height, width=width, margin=margin, title_text='Residuals per Split')
    
    st.plotly_chart(fig)

    st.subheader('Boxplot of RMSE in Cross Validation')

    fig = graphs.plot_boxplot(data=eval_df, x=None, y=None, model_name=select_model_meval, custom_feature=select_custom_features_meval, custom_target=select_custom_target_meval, single_box=True, title_text='Cross Validation with 5 Folds', height=height, width=width, margin=margin)

    st.plotly_chart(fig)

