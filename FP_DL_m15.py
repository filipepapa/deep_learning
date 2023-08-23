
#%%
# basic imports
import os, random
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.max_columns', None)
import numpy as np
import datetime as dt
import pandas_ta as ta
from pathlib import Path
from collections import Counter
# import boruta
from boruta import BorutaPy

# warnings
import warnings
warnings.filterwarnings('ignore')

# plotting & outputs
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('seaborn')
plt.style.use('default')
mpl.rcParams['figure.figsize'] = [10.0, 6.0]
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.grid'] = True

# functions from helper
# from src.tools import *

# import custom transformer
# from helper import DayTransformer, TimeTransformer

# statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pyfolio as pf


# sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



# metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import RocCurveDisplay, auc, roc_curve, plot_roc_curve


# import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest, chi2
from minisom import MiniSom


# tensorflow
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator 
from tensorflow.keras.optimizers import Adam, RMSprop 
from tensorflow.keras.losses import BinaryCrossentropy 
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy, AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.layers import LSTM, BatchNormalization

# kerastuner
import keras_tuner as kt
from kerastuner import HyperParameter, HyperParameters
from kerastuner.tuners import RandomSearch, BayesianOptimization, Hyperband

# %%
# Modified helper.py

def set_seeds(seed=42): 
    '''define seed'''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def cwts(dfs):
    '''class weight function'''
    c0, c1 = np.bincount(dfs['predict'])
    w0=(1/c0)*(len(dfs))/2 
    w1=(1/c1)*(len(dfs))/2 
    return {0: w0, 1: w1}

def plot_ft(df:pd.DataFrame,lst_plot=None,overlay=True,figsize=(20,15),lw=1,xdate=True):
    '''plot features'''
    df.dropna(inplace=True)
    if lst_plot==None:
        fig, axs = plt.subplots(df.shape[1], sharex=True, figsize=figsize) # (20,150) for all features
        if xdate:
            for i,f in enumerate(df.columns): 
                axs[i].plot(df.index, df.iloc[:,i],label=f,linewidth=lw);
                axs[i].legend(loc="lower right");
        else:
            for i,f in enumerate(df.columns): 
                axs[i].plot(list(range(df.shape[0])), df.iloc[:,i],label=f,linewidth=lw);
                axs[i].legend(loc="lower right");
    else:
        chart = df[lst_plot]
        if overlay:
            chart.plot(use_index=0)
            plt.legend(lst_plot);
        else:
            plt.subplot(2,1,1)
            df['Close'].plot(use_index=0,legend=True)
            # plt.legend('Close')
            lst_plot.remove('Close')
            plt.subplot(2,1,2)
            plt.plot(df[lst_plot].values)
            plt.legend(lst_plot);

def plot_confusion_matrix(y_test,y_pred,title=None,normalize='true',figpath=''):
    '''plot confusion matrix'''
    cf_matrix = confusion_matrix(y_test,y_pred,normalize=normalize)
    sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='.2f')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(title)
    plt.savefig(figpath);
    plt.show();

def load_def_plot():
    '''load default plot settings'''
    plt.style.use('default')
    mpl.rcParams['figure.figsize'] = [10.0, 6.0]
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['axes.grid'] = True

def plot_events(lst_events):
    '''enumerate plot events in first chart'''
    for i,s in enumerate(lst_events):
        xdt = dt.datetime.strptime(s,'%Y-%m-%d')
        plt.axvline(xdt,color='k',linestyle='--',linewidth=1)
        plt.annotate(f'({i+1})',(xdt,61000),color='k')

def fit_test_model(clf,X_train,y_train,X_test,y_test,plot_confusion=True):
    '''fit-test model and plot normalized confusion'''
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy Score in Train: \t {accuracy_score(y_train, clf.predict(X_train))*100:.4f}%")
    print(f"Accuracy Score in Test: \t {accuracy_score(y_test, y_pred)*100:.4f}%")
    if plot_confusion:
        cf_matrix = confusion_matrix(y_test,y_pred,normalize='true')
        sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='.2f')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.show();
    return y_pred

def get_features_index(sr_features:pd.Series,lst_features:list):
    '''tool to get features indices'''
    lst_scale_index = []
    for f in lst_features:
        lst_scale_index.extend(sr_features[sr_features.str[:len(f)]==f].index.tolist())
    lst_scale_index.sort()
    return list(set(lst_scale_index))

def check_duplicates(lst_scale_standard,lst_scale_minmax):
    '''check if there are features being duplicated after scaling'''
    lst_scales = []
    lst_scales.extend(lst_scale_standard)
    lst_scales.extend(lst_scale_minmax)
    lst_scales.sort()

    duplicates = [item for item, count in Counter(lst_scales).items() if count > 1]
    if duplicates:
        print('Found duplicates in ', duplicates)
    else:
        print('No duplicates found.')
    return duplicates

def plot_correlation_matrix(X,y,sr_features,compute_and_plot=True):
    '''plot correlation matrix
    -> if compute_and_plot=False corr matrix only
      will be returned'''
    Xc = X.copy()
    Xc['predict'] = y
    lst_corr = ['predict']
    lst_corr.extend(sr_features.tolist())

    df_corr = Xc[lst_corr].corr().filter(lst_corr)
    print(f'Shape of correlation matrix: {df_corr.shape}')
    # df_corr.to_excel('data/features_correlation.xlsx')
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))

    mask = mask[1:, :-1]
    corr = df_corr.iloc[1:,:-1].copy()
    
    if compute_and_plot:
        heatmap_fig, ax = plt.subplots(figsize=(20,15))
        cmap = sns.diverging_palette(0,230,90,60,as_cmap=True)

        sns.heatmap(corr, ax=ax, vmin=-.50, vmax=.50, mask=mask, square=True, linewidths=.25, cmap=cmap, cbar_kws={"shrink": .8})
        heatmap_fig.subplots_adjust(top=0.94)

    # dataframe with top correlated features to the predicting target
    df_corr_all = pd.DataFrame(corr['predict']*100)
    return df_corr_all


def create_model_1(hu=256, lookback=60, features=1):
    '''create model 2'''
    tf.keras.backend.clear_session()
    
    model = Sequential()
    
    model.add(LSTM(units=hu*2, input_shape=(lookback, features), activation = 'elu', return_sequences=True, name='LSTM_1'))
    model.add(Dropout(0.4, name='Drouput_1'))
    
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=False, name='LSTM_2'))
    
    model.add(Dense(units=1, activation='sigmoid', name='Output'))
    
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                           Precision(),
                           Recall()])
    model.summary()
    return model, 'model_1'

def create_model_2(hu=256, lookback=60, features=1):
    '''create moedl 2'''
    tf.keras.backend.clear_session()
    
    model = Sequential()
    
    model.add(LSTM(units=hu*2, input_shape=(lookback, features), activation = 'elu', return_sequences=True, name='LSTM_1'))
    model.add(Dropout(0.4, name='Drouput_1'))
    
    model.add(LSTM(units=hu*2, input_shape=(lookback, features), activation = 'elu', return_sequences=True, name='LSTM_2'))
    model.add(Dropout(0.4, name='Drouput_2'))
    
    model.add(LSTM(units=hu, activation = 'elu', return_sequences=False, name='LSTM_3'))
    
    model.add(Dense(units=1, activation='sigmoid', name='Output'))
    
    opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)
    
    # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
    model.compile(optimizer=opt, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy', 
                           Precision(),
                           Recall()])
    model.summary()
    return model, 'model_2'

def set_fit(model_name):
    '''settings to fit'''
    results_path = Path('results', 'lstm_time_series')
    if not results_path.exists():
        results_path.mkdir(parents=True)

    model_path = (results_path / f'{model_name}.h5').as_posix()
    logdir = os.path.join("./tensorboard/logs", f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}-{model_name}")

    my_callbacks = [
        EarlyStopping(patience=20, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, verbose=1, monitor='loss', save_best_only=True),
        TensorBoard(log_dir=logdir, histogram_freq=1)
    ]
    return my_callbacks

def run_fit(model,g,g_,my_callbacks,class_weight):
    '''run model fitting'''
    history = model.fit(g,
            epochs=500,
            verbose=1,
            callbacks=my_callbacks,
            shuffle=False,
            class_weight=class_weight)

    y_pred = np.where(model.predict(g_, verbose=True) > 0.5, 1, 0)

    # plot_confusion_matrix(g_.targets[seqlen:],y_pred_model_1)
    score = model.evaluate(g,verbose=0)
    print(f'Model metrics train score: {model.metrics_names[1]}, {score[1]*100:.4}%')
    score = model.evaluate(g_, verbose=0)
    print(f'Model metrics test score: {model.metrics_names[1]}, {score[1]*100:.4}%')
    return history, y_pred

# definition Class Model1
class Model1:
    def __init__(self,g,g_,seqlen,numfeat,class_weight,model_name) -> None:
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = 'model_1'
        
        self.g = g
        self.g_ = g_
        self.seqlen = seqlen
        self.numfeat = numfeat
        self.class_weight = class_weight

        pass

    def create_model(self, hu=256):
        '''create model 1'''
        
        tf.keras.backend.clear_session()
        
        self.model = Sequential()
        
        self.model.add(LSTM(units=hu*2, input_shape=(self.seqlen, self.numfeat), activation = 'elu', return_sequences=True, name='LSTM_1'))
        self.model.add(Dropout(0.4, name='Drouput_1'))
        
        self.model.add(LSTM(units=hu, activation = 'elu', return_sequences=False, name='LSTM_2'))
        
        self.model.add(Dense(units=1, activation='sigmoid', name='Output'))
        
        opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)
        
        self.model.compile(optimizer=opt, 
                    loss=BinaryCrossentropy(), 
                    metrics=['accuracy', 
                            Precision(),
                            Recall()])
        self.model.summary()

        return self.model
    
    def set_fit(self, es_patience=20, es_monitor='loss'):
        '''settings to fit'''
        self.es_monitor = es_monitor
        results_path = Path('results', 'lstm_time_series')
        if not results_path.exists():
            results_path.mkdir(parents=True)

        model_path = (results_path / f'{self.model_name}.h5').as_posix()
        
        logdir = os.path.join("./tensorboard/logs", f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}-{self.model_name}")
        
        mode= 'max' if 'val' in es_monitor else 'min'

        self.my_callbacks = [
                EarlyStopping(patience=es_patience, monitor=es_monitor, mode=mode, verbose=1, restore_best_weights=True),
                ModelCheckpoint(filepath=model_path, verbose=1, monitor=es_monitor, save_best_only=True),
                TensorBoard(log_dir=logdir, histogram_freq=1)
        ]

        print(f'Settings model_path: {model_path}')
        print(f'Settings logdir: {logdir}')
        return

    # Run Fit algorithm
    def run_fit(self):
        '''run model fitting'''
        self.history = self.model.fit(self.g,
                                    epochs=500,
                                    verbose=1,
                                    validation_data=self.g_,
                                    callbacks=self.my_callbacks,
                                    shuffle=False,
                                    class_weight=self.class_weight)
        
        self.y_prob = self.model.predict(self.g_, verbose=True)
        self.y_pred = np.where(self.y_prob > 0.5, 1, 0)

        score = self.model.evaluate(self.g,verbose=0)
        print(f'Model metrics train score: {self.model.metrics_names[1]}, {score[1]*100:.4}%')
        score = self.model.evaluate(self.g_, verbose=0)
        print(f'Model metrics test score: {self.model.metrics_names[1]}, {score[1]*100:.4}%')

        return self.history, self.y_pred

    def hpo_model(self,hp):
        '''hyperparameter opt build model 1'''
        tf.keras.backend.clear_session()   

        self.model = Sequential()
        
        # neurons in LSTM
        hp_units1 = hp.Int('units1', min_value=8, max_value=36, step=2)
        hp_units2 = hp.Int('units2', min_value=8, max_value=36, step=2)
        
        # dropout rate
        hp_dropout1 = hp.Float('Dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

        # learning rate for the optimizer
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # activation functions
        hp_activation1 = hp.Choice(name = 'activation', values = ['relu','elu','sigmoid','tanh'], ordered = False)
        hp_activation2 = hp.Choice(name = 'activation', values = ['relu','elu','sigmoid','tanh'], ordered = False)
        

        # model_1
        self.model.add(LSTM(hp_units1, input_shape=(self.seqlen, self.numfeat), activation=hp_activation1, return_sequences=True, name='LSTM_1'))
        self.model.add(Dropout(hp_dropout1, name='Drouput_1'))
        
        self.model.add(LSTM(hp_units2, activation = hp_activation2, return_sequences=False, name='LSTM_2'))
        
        self.model.add(Dense(units=1, activation='sigmoid', name='Output'))

        # specify optimizer separately (preferred method))
        opt = Adam(lr=hp_learning_rate, epsilon=1e-08, decay=0.0)       
        
        # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
        self.model.compile(optimizer=opt, 
                    loss=BinaryCrossentropy(), 
                    metrics=['accuracy', 
                            Precision(),
                            Recall()])

        # set callbacks
        self.callback_hp_rs = [EarlyStopping(patience=5, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
                    TensorBoard(log_dir=f"./tensorboard/rslogs-{self.model_name}")]

        self.callback_hp_hb = [EarlyStopping(patience=5, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
                    TensorBoard(log_dir=f"./tensorboard/hblogs-{self.model_name}")]
        
        return self.model

    def create_model_best_hp(self, tuner_best):
        '''generate best hp model'''
        self.tuner_best = tuner_best
        
        tf.keras.backend.clear_session()

        self.model = Sequential()
        
        self.model.add(LSTM(units=self.tuner_best['units1'], input_shape=(self.seqlen, self.numfeat), activation = self.tuner_best['activation'], return_sequences=True, name='LSTM_1'))
        self.model.add(Dropout(self.tuner_best['Dropout_rate'], name='Drouput_1'))
        
        self.model.add(LSTM(units=self.tuner_best['units2'], activation = self.tuner_best['activation'], return_sequences=False, name='LSTM_2'))
        
        self.model.add(Dense(units=1, activation='sigmoid', name='Output'))
        
        opt = Adam(lr=self.tuner_best['learning_rate'], epsilon=1e-08, decay=0.0)
        
        self.model.compile(optimizer=opt, 
                    loss=BinaryCrossentropy(), 
                    metrics=['accuracy', 
                            Precision(),
                            Recall()])
        self.model.summary()

        return self.model
    
    def plot_acc_loss(self,title=None,plot_val=False):
        '''plot accuracy and loss subplot(2,1)'''
        if not plot_val:
            sr_train_acc = pd.Series(self.history.history['accuracy'],name='Model 1 Train Accuracy')
            sr_train_loss = pd.Series(self.history.history['loss'],name='Model 1 Train Loss')
            
            fig, axs = plt.subplots(2,1)
            axs[0] = plt.subplot(2,1,1)
            sr_train_acc.plot(legend=True)
            axs[1] = plt.subplot(2,1,2,sharex=axs[0])
            sr_train_loss.plot(legend=True)
        else:
            sr_train_acc = pd.Series(self.history.history['accuracy'],name='Model 1 Train Accuracy')
            sr_train_loss = pd.Series(self.history.history['loss'],name='Model 1 Train Loss')
            sr_val_acc = pd.Series(self.history.history['val_accuracy'],name='Model 1 Validation Accuracy')
            sr_val_loss = pd.Series(self.history.history['val_loss'],name='Model 1 Validation Loss')
            
            fig, axs = plt.subplots(2,1)
            axs[0] = plt.subplot(2,1,1)
            sr_train_acc.plot(legend=True)
            sr_val_acc.plot(legend=True)
            axs[1] = plt.subplot(2,1,2,sharex=axs[0])
            sr_train_loss.plot(legend=True)
            sr_val_loss.plot(legend=True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        if title:
            fig.suptitle(title);

# definition Class Model2
class Model2:
    def __init__(self,g,g_,seqlen,numfeat,class_weight,model_name=None) -> None:
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = 'model_2'
        self.g = g
        self.g_ = g_
        self.seqlen = seqlen
        self.numfeat = numfeat
        self.class_weight = class_weight

        pass

    def create_model(self, hu=256):
        '''create model 1'''
        tf.keras.backend.clear_session()
        
        self.model = Sequential()
        
        self.model.add(LSTM(units=hu*2, input_shape=(self.seqlen, self.numfeat), activation = 'elu', return_sequences=True, name='LSTM_1'))
        self.model.add(Dropout(0.4, name='Drouput_1'))
        
        self.model.add(LSTM(units=hu*2, input_shape=(self.seqlen, self.numfeat), activation = 'elu', return_sequences=True, name='LSTM_2'))
        self.model.add(Dropout(0.4, name='Drouput_2'))
        
        self.model.add(LSTM(units=hu, activation = 'elu', return_sequences=False, name='LSTM_3'))
        
        self.model.add(Dense(units=1, activation='sigmoid', name='Output'))
        
        opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)
        
        self.model.compile(optimizer=opt, 
                    loss=BinaryCrossentropy(), 
                    metrics=['accuracy', 
                            Precision(),
                            Recall()])
        self.model.summary()

        return self.model
    
    def set_fit(self,patience=20, es_monitor='loss'):
        '''settings to fit'''
        self.patience = patience
        self.es_monitor = es_monitor

        results_path = Path('results', 'lstm_time_series')
        if not results_path.exists():
            results_path.mkdir(parents=True)

        model_path = (results_path / f'{self.model_name}.h5').as_posix()
        logdir = os.path.join("./tensorboard/logs", f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}-{self.model_name}")

        mode= 'max' if 'val' in es_monitor else 'min'

        self.my_callbacks = [
            EarlyStopping(patience=self.patience, monitor=self.es_monitor, mode=mode, verbose=1, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, verbose=1, monitor=self.es_monitor, save_best_only=True),
            TensorBoard(log_dir=logdir, histogram_freq=1)
        ]
        print(f'Settings model_path: {model_path}')
        print(f'Settings logdir: {logdir}')
        return

    def run_fit(self):
        '''run model fitting'''
        self.history = self.model.fit(self.g,
                                    epochs=500,
                                    verbose=1,
                                    validation_data=self.g_,
                                    callbacks=self.my_callbacks,
                                    shuffle=False,
                                    class_weight=self.class_weight)

        self.y_prob = self.model.predict(self.g_, verbose=True)
        self.y_pred = np.where( self.y_prob > 0.5, 1, 0)

        score = self.model.evaluate(self.g,verbose=0)
        print(f'Model metrics train score: {self.model.metrics_names[1]}, {score[1]*100:.4}%')
        score = self.model.evaluate(self.g_, verbose=0)
        print(f'Model metrics test score: {self.model.metrics_names[1]}, {score[1]*100:.4}%')

        return self.history, self.y_pred

    def hpo_model(self,hp):
        '''hyperparameter opt build model 1'''
        
        tf.keras.backend.clear_session()   

        self.model = Sequential()
        
        # neurons in LSTM
        hp_units1 = hp.Int('units1', min_value=8, max_value=36, step=2)
        hp_units2 = hp.Int('units2', min_value=8, max_value=36, step=2)
        hp_units3 = hp.Int('units3', min_value=8, max_value=36, step=2)
        
        # dropout rate
        hp_dropout1 = hp.Float('Dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        hp_dropout2 = hp.Float('Dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

        # learning rate for the optimizer
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # activation functions
        hp_activation1 = hp.Choice(name = 'activation', values = ['relu','elu','sigmoid','tanh'], ordered = False)
        hp_activation2 = hp.Choice(name = 'activation', values = ['relu','elu','sigmoid','tanh'], ordered = False)
        hp_activation3 = hp.Choice(name = 'activation', values = ['relu','elu','sigmoid','tanh'], ordered = False)
        

        # model_1
        self.model.add(LSTM(hp_units1, input_shape=(self.seqlen, self.numfeat), activation=hp_activation1, return_sequences=True, name='LSTM_1'))
        self.model.add(Dropout(hp_dropout1, name='Drouput_1'))
        
        self.model.add(LSTM(hp_units2, input_shape=(self.seqlen, self.numfeat), activation=hp_activation2, return_sequences=True, name='LSTM_2'))
        self.model.add(Dropout(hp_dropout2, name='Drouput_2'))
        
        self.model.add(LSTM(hp_units3, activation = hp_activation3, return_sequences=False, name='LSTM_3'))
        
        self.model.add(Dense(units=1, activation='sigmoid', name='Output'))

        # specify optimizer separately (preferred method))
        opt = Adam(lr=hp_learning_rate, epsilon=1e-08, decay=0.0)       
        
        # model compilation - 'binary_crossentropy' - 'accuracy' - BinaryAccuracy(name='accuracy', threshold=0.5)
        self.model.compile(optimizer=opt, 
                    loss=BinaryCrossentropy(), 
                    metrics=['accuracy', 
                            Precision(),
                            Recall()])

        # set callbacks
        self.callback_hp_rs = [EarlyStopping(patience=5, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
                    TensorBoard(log_dir=f"./tensorboard/rslogs-{self.model_name}")]

        self.callback_hp_hb = [EarlyStopping(patience=5, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
                    TensorBoard(log_dir=f"./tensorboard/hblogs-{self.model_name}")]
        
        return self.model

    def create_model_best_hp(self, tuner_best):
        '''generate best hp model'''
        self.tuner_best = tuner_best
        
        tf.keras.backend.clear_session()

        self.model = Sequential()
        
        self.model.add(LSTM(units=self.tuner_best['units1'], input_shape=(self.seqlen, self.numfeat), activation = self.tuner_best['activation'], return_sequences=True, name='LSTM_1'))
        self.model.add(Dropout(self.tuner_best['Dropout_rate'], name='Drouput_1'))
        
        self.model.add(LSTM(units=self.tuner_best['units2'], input_shape=(self.seqlen, self.numfeat), activation = self.tuner_best['activation'], return_sequences=True, name='LSTM_2'))
        self.model.add(Dropout(self.tuner_best['Dropout_rate'], name='Drouput_2'))
        
        self.model.add(LSTM(units=self.tuner_best['units3'], activation = self.tuner_best['activation'], return_sequences=False, name='LSTM_3'))
        
        self.model.add(Dense(units=1, activation='sigmoid', name='Output'))
        
        opt = Adam(lr=self.tuner_best['learning_rate'], epsilon=1e-08, decay=0.0)
        
        self.model.compile(optimizer=opt, 
                    loss=BinaryCrossentropy(),
                    metrics=['accuracy', 
                            Precision(),
                            Recall()])
        self.model.summary()

        return self.model
    
    def plot_acc_loss(self,title=None,plot_val=False):
        '''plot accuracy and loss subplot(2,1)'''
        if not plot_val:
            sr_train_acc = pd.Series(self.history.history['accuracy'],name='Model 2 Train Accuracy')
            sr_train_loss = pd.Series(self.history.history['loss'],name='Model 2 Train Loss')
            
            fig, axs = plt.subplots(2,1)
            axs[0] = plt.subplot(2,1,1)
            sr_train_acc.plot(legend=True)
            axs[1] = plt.subplot(2,1,2,sharex=axs[0])
            sr_train_loss.plot(legend=True)

        else:
            sr_train_acc = pd.Series(self.history.history['accuracy'],name='Model 2 Train Accuracy')
            sr_train_loss = pd.Series(self.history.history['loss'],name='Model 2 Train Loss')
            sr_val_acc = pd.Series(self.history.history['val_accuracy'],name='Model 2 Validation Accuracy')
            sr_val_loss = pd.Series(self.history.history['val_loss'],name='Model 2 Validation Loss')
            
            fig, axs = plt.subplots(2,1)
            axs[0] = plt.subplot(2,1,1)
            sr_train_acc.plot(legend=True)
            sr_val_acc.plot(legend=True)
            axs[1] = plt.subplot(2,1,2,sharex=axs[0])
            sr_train_loss.plot(legend=True)
            sr_val_loss.plot(legend=True)

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        if title:
            fig.suptitle(title);

def compute_underwater(sr_ret:pd.Series):
    df_cum_rets = sr_ret.cumsum().apply(np.exp)
    running_max = np.maximum.accumulate(df_cum_rets)
    sr_underwater = -100 * ((running_max - df_cum_rets) / running_max)
    return sr_underwater
#%%
ticker = 'IBOV_M15'
fpath = rf'data\{ticker}.csv'
df = pd.read_csv(fpath,sep=';',parse_dates=['Date'],index_col=0).sort_index()
df

# %%
### Feature Extraction
# add all factors
df_f = df.copy()
# df_f.ta.strategy('All')
CustomStrategy = ta.Strategy(
    name="Deep Learning Features",
    description="Main technical indices proposed by experience and aiming better fitting for appropriate transformation by technical indices categories",
    ta=[
        # robust scaler
        {"kind": "log_return", "length": 1},
        {"kind": "log_return", "length": 2},
        {"kind": "log_return", "length": 3},
        {"kind": "log_return", "length": 4},
        {"kind": "log_return", "length": 5},
        {"kind": "percent_return", "length": 1},
        {"kind": "percent_return", "length": 2},
        {"kind": "percent_return", "length": 3},
        {"kind": "percent_return", "length": 4},
        {"kind": "percent_return", "length": 5},
        
        # robust scaler
        {"kind": "mom", "length": 1},
        {"kind": "mom", "length": 2},
        {"kind": "mom", "length": 3},
        {"kind": "mom", "length": 4},
        {"kind": "mom", "length": 5},

        # minmax scaler
        {"kind": "sma", "length": 9},
        {"kind": "sma", "length": 18},
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 100},
        {"kind": "sma", "length": 150},
        {"kind": "sma", "length": 200},

        {"kind": "ema", "length": 9},
        {"kind": "ema", "length": 18},
        {"kind": "ema", "length": 50},
        {"kind": "ema", "length": 100},
        {"kind": "ema", "length": 150},
        {"kind": "ema", "length": 200},
        
        # minmax scaler
        {"kind": "adx"},
        {"kind": "stochrsi"},
        {"kind": "obv"},
        {"kind": "vwap"},
        
        # standard scaler
        {"kind": "atr"},
        {"kind": "trix"},
        {"kind": "bbands",  "col_names": ("BBL_5_2", "BBM_5_2", "BBU_5_2", "BBB_5_2", "BBP_5_2")},
        {"kind": "macd",    "col_names": ("MACD_12_26_9", "MACD_H_12_26_9", "MACD_S_12_26_9")}
    ]
)

df_f.ta.strategy(CustomStrategy,timed=True,verbose=True)
lst_features = df_f.columns.to_list()
s_features = pd.Series(lst_features)
print(f'Shape after features extraction: {s_features.shape}')
# %%
### Expanded Features
# enhancing features
# BBands Signal Feature
# df_f['BBs_5_2.0'] = (df_f['Close']-df_f['BBL_5_2.0']) / (df_f['BBU_5_2.0']-df_f['BBL_5_2.0'])

# DMI Signal Feature
df_f['DMI_14'] = df_f['ADX_14'] * (df_f['DMP_14']-df_f['DMN_14'])

# Stochastic RSI Signal Feature
df_f['STOCHs_14_14_3_3'] = df_f['STOCHRSIk_14_14_3_3'] - df_f['STOCHRSId_14_14_3_3']

# TRIX Histogram Feature
df_f['TRIXh_30_9'] = df_f['TRIX_30_9'] - df_f['TRIXs_30_9']

# OBV Moving Average Signal Features
for i in [5,20,50,100,200,500]:
    df_f[f'OBV_MA_{i}'] = df_f['OBV'].rolling(i).mean()
    df_f[f'OBVs_{i}'] = df_f['OBV'] - df_f[f'OBV_MA_{i}']

# C-VWAP_D Signal Features
for i in [5,20,50,100,200,500]:
    df_f[f'C-VWAP_D_{i}'] = df_f['Close'] - df_f[f'VWAP_D'].shift(i)

# C-H Signal Features
for i in [5,20,50,100,200,500]:
    df_f[f'C-H_{i}'] = df_f['Close'] - df_f['High'].rolling(i).max()

# C-L Signal Features
for i in [5,20,50,100,200,500]:
    df_f[f'C-L_{i}'] = df_f['Close'] - df_f['Low'].rolling(i).min()

# C-SMA Signal Features
sma_features = s_features[s_features.str[:3]=='SMA'].tolist()
for sma in sma_features:
    df_f[f'C-{sma}'] = df_f['Close'] - df_f[sma]

# C-EMA Signal Features
ema_features = s_features[s_features.str[:3]=='EMA'].tolist()
for ema in ema_features:
    df_f[f'C-{ema}'] = df_f['Close'] - df_f[ema]

### Original and Expanded Features
# all features series
lst_features = df_f.columns.to_list()
s_features = pd.Series(lst_features)
s_features.shape
print(f'Shape after features enhancing: {s_features.shape}')
# %%
### Data Handling: Target Specification
data = df_f.copy().dropna()

hpredict = 4
data['predict'] = np.where(data['Close'].pct_change(hpredict).shift(-hpredict) > 0.00/100, 1, 0) # TBD
data[f'predict_PCTRET_{hpredict}'] = data['Close'].pct_change(hpredict).shift(-hpredict)

data = data[:-hpredict] # for 5h prediction
# check for missing values
print(f'Current na values in dataset: {data.isna().sum().sum()}')

# check last 10 rows (returns in %)
cols = ['Close', 'PCTRET_5', 'predict']
print(data[cols].tail(20) * np.array([1,100,1]))

## Handling Class Imbalances
class_weight = cwts(data)
c0, c1 = np.bincount(data['predict'])
print()
print('Class Weights')
print(f'c0: {c0}, c1: {c1}')
print(f'w0: {class_weight[0]:.4f}, w1: {class_weight[1]:.4f}')
# %%
### Split Data
X = data.drop(['predict', f'predict_PCTRET_{hpredict}','Open','High','Low','Close','VWAP_D','BBL_5_2','BBM_5_2','BBU_5_2'], axis=1)
feature_names = X.columns.tolist()
s_features = pd.Series(lst_features)

# Label Definition
y = data['predict'].values
y = y.astype(int)

# Split Data
df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train, X_test, y_train, y_test = np.array(df_X_train), np.array(df_X_test), np.array(df_y_train), np.array(df_y_test) 

# Output the train and test data size
print(f"Train and Test Size {len(X_train)}, {len(X_test)}")
# %%
### Random Forest Classifier Test
# define random forest classifier
forest = RandomForestClassifier(n_jobs=-1, 
                                class_weight=cwts(data), 
                                random_state=42, 
                                max_depth=3)

# train the model
y_pred = fit_test_model(forest,X_train,y_train,X_test,y_test)
# %%
### Data Handling
# lst_features_minmax = ['OBVs_5','OBVs_20','OBVs_50','OBVs_100','OBVs_200','OBVs_500','DMP_14','DMN_14','STOCHRSIk_14_14_3_3','STOCHRSId_14_14_3_3','STOCHs_14_14_3_3','BBP_5_2']
lst_features_stdsca = ['Volume','LOGRET','PCTRET','OBV','OBV_MA','SMA','EMA','MOM','C-SMA','C-EMA','C-H','C-L','C-VWAP_D','ADX','DMI','TRIX','TRIXs','TRIXh','MACD','MACD_H','MACD_S','ATRr','BBB']
lst_features_minmax = ['OBVs','DMP','DMN','STOCHRSIk','STOCHRSId','STOCHs','BBP']

sr_features = pd.Series(df_X_train.columns)
lst_scale_standard = get_features_index(sr_features,lst_features_stdsca)
lst_scale_minmax = get_features_index(sr_features,lst_features_minmax)


duplicates = check_duplicates(lst_scale_standard,lst_scale_minmax)
# handle duplicates (removing from undesired scaler)
print('Removing duplicates..')
lst_scale_standard = [f for f in lst_scale_standard if f not in duplicates] # removing OBVs duplicates from standard
duplicates = check_duplicates(lst_scale_standard,lst_scale_minmax)

# %%
### Scaling Core Features
ct = ColumnTransformer(
    transformers=[
        ('standard', StandardScaler(), lst_scale_standard),
        ('minmax', MinMaxScaler(), lst_scale_minmax)
    ])
X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

df_X_train_scaled = pd.DataFrame(X_train_scaled,columns=sr_features)
df_X_test_scaled = pd.DataFrame(X_test_scaled,columns=sr_features)

df_X_train_minmax = pd.DataFrame(X_train_minmax,columns=sr_features)
df_X_test_minmax = pd.DataFrame(X_test_minmax,columns=sr_features)

df_X_train_std = pd.DataFrame(X_train_std,columns=sr_features)
df_X_test_std = pd.DataFrame(X_test_std,columns=sr_features)

# %% 
# PCA Transform
pca = PCA(n_components=20)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
X_train_pca_scaled = pca.fit_transform(X_train_scaled)
X_test_pca_scaled = pca.transform(X_test_scaled)
X_train_pca_minmax = pca.fit_transform(X_train_minmax)
X_test_pca_minmax = pca.transform(X_test_minmax)
X_train_pca_std = pca.fit_transform(X_train_std)
X_test_pca_std = pca.transform(X_test_std)
#%%
# PCA % of variance explained
df_pca_varexp = pd.DataFrame(np.cumsum(pca.explained_variance_ratio_)*100,columns=['% of Explained Variance'])
df_pca_varrat = pd.DataFrame(pca.explained_variance_ratio_*100,columns=['Explained Variance Ratio (%)'])

load_def_plot()
fig,ax = plt.subplots()
df_pca_varexp.plot(kind='line',ax=ax)
df_pca_varrat.plot(kind='bar',ax=ax,color='green',alpha=.3,sharex=ax)
plt.xlabel('Number of Principal Components')
plt.savefig(f'images/pca_varexp.png');
# %% 
### Boruta feature selection Method
forest = RandomForestClassifier(n_jobs=-1, 
                                class_weight=cwts(data), 
                                random_state=42, 
                                max_depth=3)
feat_selector = BorutaPy(forest, 
                         verbose=2, 
                         random_state=42)

# find all relevant features
# takes input in array format not as dataframe
print('Running with no scaler:')
print('Fitting BorutaPy')
feat_selector.fit(X_train, y_train)
# call transform() on X to filter it down to selected features
X_train_boruta = feat_selector.transform(X_train)
X_test_boruta = feat_selector.transform(X_test)

# filtering features in scaled arrays
X_train_boruta_std = X_train_std[:,feat_selector.support_]
X_test_boruta_std = X_test_std[:,feat_selector.support_]
X_train_boruta_scaled = X_train_scaled[:,feat_selector.support_]
X_test_boruta_scaled = X_test_scaled[:,feat_selector.support_]


# %% 
# TimeSeries Generator
seqlen = 21

# number of features
g = TimeseriesGenerator(X_train, y_train, length=seqlen)
g_ = TimeseriesGenerator(X_test, y_test, length=seqlen)
g_scaled = TimeseriesGenerator(X_train_scaled, y_train, length=seqlen)
g_scaled_ = TimeseriesGenerator(X_test_scaled, y_test, length=seqlen)
g_minmax = TimeseriesGenerator(X_train_minmax, y_train, length=seqlen)
g_minmax_ = TimeseriesGenerator(X_test_minmax, y_test, length=seqlen)
g_std = TimeseriesGenerator(X_train_std, y_train, length=seqlen)
g_std_ = TimeseriesGenerator(X_test_std, y_test, length=seqlen)
g_pca = TimeseriesGenerator(X_train_pca, y_train, length=seqlen)
g_pca_ = TimeseriesGenerator(X_test_pca, y_test, length=seqlen)
g_pca_minmax = TimeseriesGenerator(X_train_pca_minmax, y_train, length=seqlen)
g_pca_minmax_ = TimeseriesGenerator(X_test_pca_minmax, y_test, length=seqlen)
g_pca_std = TimeseriesGenerator(X_train_pca_std, y_train, length=seqlen)
g_pca_std_ = TimeseriesGenerator(X_test_pca_std, y_test, length=seqlen)
g_boruta_std = TimeseriesGenerator(X_train_boruta_std, y_train, length=seqlen)
g_boruta_std_ = TimeseriesGenerator(X_test_boruta_std, y_test, length=seqlen)
g_boruta_scaled = TimeseriesGenerator(X_train_boruta_scaled, y_train, length=seqlen)
g_boruta_scaled_ = TimeseriesGenerator(X_test_boruta_scaled, y_test, length=seqlen)

#%%
dg = {}
dg_ = {}
dg['vanilla'] = TimeseriesGenerator(X_train, y_train, length=seqlen)
dg_['vanilla'] = TimeseriesGenerator(X_test, y_test, length=seqlen)
dg['scaled'] = TimeseriesGenerator(X_train_scaled, y_train, length=seqlen)
dg_['scaled'] = TimeseriesGenerator(X_test_scaled, y_test, length=seqlen)
dg['minmax'] = TimeseriesGenerator(X_train_minmax, y_train, length=seqlen)
dg_['minmax'] = TimeseriesGenerator(X_test_minmax, y_test, length=seqlen)
dg['std'] = TimeseriesGenerator(X_train_std, y_train, length=seqlen)
dg_['std'] = TimeseriesGenerator(X_test_std, y_test, length=seqlen)
dg['pca'] = TimeseriesGenerator(X_train_pca, y_train, length=seqlen)
dg_['pca'] = TimeseriesGenerator(X_test_pca, y_test, length=seqlen)
dg['pca_minmax'] = TimeseriesGenerator(X_train_pca_minmax, y_train, length=seqlen)
dg_['pca_minmax'] = TimeseriesGenerator(X_test_pca_minmax, y_test, length=seqlen)
dg['pca_std'] = TimeseriesGenerator(X_train_pca_std, y_train, length=seqlen)
dg_['pca_std'] = TimeseriesGenerator(X_test_pca_std, y_test, length=seqlen)
dg['boruta_std'] = TimeseriesGenerator(X_train_boruta_std, y_train, length=seqlen)
dg_['boruta_std'] = TimeseriesGenerator(X_test_boruta_std, y_test, length=seqlen)
dg['boruta_scaled'] = TimeseriesGenerator(X_train_boruta_scaled, y_train, length=seqlen)
dg_['boruta_scaled'] = TimeseriesGenerator(X_test_boruta_scaled, y_test, length=seqlen)

# %% 
# Best Hyperparameters
hbtuner_2_best = {'units1': 12,
                        'units2': 24,
                        'units3': 20,
                        'Dropout_rate': 0.1,
                        'learning_rate': 0.001,
                        'activation': 'relu',
                        'tuner/epochs': 5,
                        'tuner/initial_epoch': 2,
                        'tuner/bracket': 1,
                        'tuner/round': 1,
                        'tuner/trial_id': 'e3c7df02436ed1a6180ccfdfcbab31fe'}
m2 = {}
dhistory = {}
dypred = {}
#%%
# Running Model 2 PCA:
# useless, since it's not scaled produces too much loss and not train/val acc improvements

# monitor = 'loss'
# numfeat = g_pca.data.shape[1] #scale minmax in pca

# model_2_pca = Model2(g_pca,g_pca_,seqlen,numfeat,class_weight,model_name=f'model_2_pca_{monitor}')
# model_2_pca.set_fit(es_monitor=monitor)

# model_2_pca.create_model_best_hp(hbtuner_2_best)

# history_2_pca, y_pred_2_pca = model_2_pca.run_fit()

# model_2_pca.plot_acc_loss(model_2_pca.model_name,plot_val=True)
#%%
# Running Model 2 PCA MinMax: 
# produces best improvements in both train and val acc - a good approach can be using n_components = 15
monitor = 'loss'
mname = 'pca_minmax'
mtype = 'model2'
obs = 'pca15'
model_name = f'{mtype}_{mname}_{monitor}' if obs=='' else f'{mtype}_{mname}_{obs}_{monitor}'
print(f'model: {model_name}')
print(f'descr: {obs}')

numfeat = dg[mname].data.shape[1]

m2[mname] = Model2(dg[mname],dg_[mname],seqlen,numfeat,class_weight,model_name=model_name)

m2[mname].set_fit(es_monitor=monitor)

m2[mname].create_model_best_hp(hbtuner_2_best)

history_2_boruta_std, y_pred_2_boruta_std = m2[mname].run_fit()

m2[mname].plot_acc_loss(model_name,plot_val=True)

#%%
# Running Model 2 PCA Standard Scaler: 
# train acc more smoothed (better than minmax) but val loss and acc not good (random - feature selection)
# still worth more tests
monitor = 'loss'
numfeat = g_pca_std.data.shape[1]

model_2_pca_std = Model2(g_pca_std,g_pca_std_,seqlen,numfeat,class_weight,model_name=f'model_2_pca10_std_{monitor}')
model_2_pca_std.set_fit(es_monitor=monitor)

model_2_pca_std.create_model_best_hp(hbtuner_2_best)

history_2_pca_std, y_pred_2_pca_std = model_2_pca_std.run_fit()

model_2_pca_std.plot_acc_loss(model_2_pca_std.model_name,plot_val=True)

#%%
# Running Model 2 MinMax: 
# creates uncorrelated data to train and test - not worth
# monitor = 'loss'
# numfeat = g_minmax.data.shape[1]

# model_2_minmax = Model2(g_minmax,g_minmax_,seqlen,numfeat,class_weight,model_name=f'model_2_minmax_{monitor}')
# model_2_minmax.set_fit(es_monitor=monitor)

# model_2_minmax.create_model_best_hp(hbtuner_2_best)

# history_2_minmax, y_pred_2_minmax = model_2_minmax.run_fit()

# model_2_minmax.plot_acc_loss(model_2_minmax.model_name,plot_val=True)
# %%
# Running Model 2 StandardScaler:
# train acc more smoothed (better than minmax) but val loss and acc not good (random - feature selection)
# still worth more tests
monitor = 'loss'
numfeat = g_std.data.shape[1]

model_2_std = Model2(g_std,g_std_,seqlen,numfeat,class_weight,model_name=f'model_2_std_{monitor}')
model_2_std.set_fit(es_monitor=monitor)

model_2_std.create_model_best_hp(hbtuner_2_best)

history_2_std, y_pred_2_std = model_2_std.run_fit()

model_2_std.plot_acc_loss(model_2_std.model_name,plot_val=True)
# %%
# Running Model 2 Scaled (Custom): 
# quite better in comparison to standard scaler, however improvements can be made to improve val acc and loss. 
# still, not better than pca minmax
monitor = 'loss'
numfeat = g_scaled.data.shape[1]

model_2_scaled = Model2(g_scaled,g_scaled_,seqlen,numfeat,class_weight,model_name=f'model_2_scaled_{monitor}')
model_2_scaled.set_fit(es_monitor=monitor)

model_2_scaled.create_model_best_hp(hbtuner_2_best)

history_2_scaled, y_pred_2_scaled = model_2_scaled.run_fit()

model_2_scaled.plot_acc_loss(model_2_scaled.model_name,plot_val=True)
# %%
# Running Model 2 Boruta Std
monitor = 'loss'
mname = 'boruta_std'
mtype = 'model2'
# obs = 'pca_15'
model_name = f'{mtype}_{mname}_{monitor}'
print(f'model: {model_name}')
# print(f'descr: {obs}')

numfeat = dg[mname].data.shape[1]

m2[mname] = Model2(dg[mname],dg_[mname],seqlen,numfeat,class_weight,model_name=model_name)

m2[mname].set_fit(es_monitor=monitor)

m2[mname].create_model_best_hp(hbtuner_2_best)

dhistory[model_name], dypred[model_name] = m2[mname].run_fit()

m2[mname].plot_acc_loss(model_name,plot_val=True)
# %%
# Running Model 2 Boruta Scaled Custom
monitor = 'loss'
mname = 'boruta_scaled'
mtype = 'model2'
# obs = 'pca_15'
model_name = f'{mtype}_{mname}_{monitor}'
print(f'model: {model_name}')
# print(f'descr: {obs}')

numfeat = dg[mname].data.shape[1]

m2[mname] = Model2(dg[mname],dg_[mname],seqlen,numfeat,class_weight,model_name=model_name)

m2[mname].set_fit(es_monitor=monitor)

m2[mname].create_model_best_hp(hbtuner_2_best)

dhistory[model_name], dypred[model_name] = m2[mname].run_fit()

m2[mname].plot_acc_loss(model_name,plot_val=True)
