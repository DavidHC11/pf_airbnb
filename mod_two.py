import pandas as pd
import numpy as np
from datetime import datetime
import datetime
import time
import math
import matplotlib
import matplotlib.pyplot as plt
from plotly.offline import plot,iplot
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
import plotly.express as px
import cufflinks as cf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,StratifiedKFold
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from  scipy.stats  import  shapiro 
from sklearn.linear_model import Lasso
from numpy import arange
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV
import pickle







def km(df,lat1, lon1, lat2, lon2):
    aux_1=df.copy()
    for x in [lat1, lon1, lat2, lon2]:
        aux_1[f'{x}rad']=aux_1[x].map(lambda x: math.radians(x))
        
    aux_1['a_lt1']=aux_1[f'{lat1}rad'].map(lambda x: math.sin(x))
    aux_1['a_lt2']=aux_1[f'{lat2}rad'].map(lambda x: math.sin(x))
    aux_1['a']=aux_1.a_lt1 * aux_1.a_lt2
    aux_1['b_lt1']=aux_1[f'{lat1}rad'].map(lambda x: math.cos(x))
    aux_1['b_lt2']=aux_1[f'{lat2}rad'].map(lambda x: math.cos(x))   
    aux_1['lg2_lg1']=aux_1[f'{lon2}rad'] - aux_1[f'{lon1}rad']
    aux_1['lg2_lg1']=aux_1['lg2_lg1'].map(lambda x : math.cos(x))
    aux_1['b']=aux_1.b_lt1 * aux_1.b_lt2 * aux_1.lg2_lg1
    aux_1['d'] = aux_1.a + aux_1.b
    aux_1['d'] = aux_1['d'].map(lambda x : math.acos(x))
    aux_1['distance']= aux_1['d'].map(lambda x: 111.18 * math.degrees(x))
    aux_1['distance']=aux_1['distance'].map(lambda x: round(x,2))
    
    return aux_1['distance']

def count_generator(df):
    for column in df.columns:
        yield df[column].value_counts(1)
        
def grip_search(X_train,y_train,estimator,param_grid):
    grid = GridSearchCV(cv=StratifiedKFold(10),
                  verbose=True,
                  scoring='r2',
                  estimator=estimator,
                  n_jobs=-1,
                  param_grid=param_grid)
    grid.fit(X_train,y_train)
    print(f"Best Score : {grid.best_score_}")
    print(f"Best Params : {grid.best_params_}")
    return grid.best_estimator_


def metricas(modelo,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test):
    ms=pd.DataFrame(columns=('modelo','datos','r2','r2_adj','mae','mse','rmse'))
    ms.loc[len(ms)]=list([modelo,'train',r2_score(y_train, y_pred_train),(1-(((df_train.shape[0]-1)/(df_train.shape[0]-df_train.shape[1]-1)))*(1-r2_score(y_train, y_pred_train))),mean_absolute_error(y_train,y_pred_train),mean_squared_error(y_train,y_pred_train),np.sqrt(mean_squared_error(y_train,y_pred_train))])
    ms.loc[len(ms)]=list([modelo,'test',r2_score(y_test, y_pred_test),(1-(((df_test.shape[0]-1)/(df_test.shape[0]-df_test.shape[1]-1)))*(1-r2_score(y_test, y_pred_test))),mean_absolute_error(y_test,y_pred_test),mean_squared_error(y_test,y_pred_test),np.sqrt(mean_squared_error(y_test,y_pred_test))])
    return ms

def reglin(df_train, y_train, df_test):
    model=LinearRegression()
    model.fit(df_train,y_train)
    y_pred_test=model.predict(df_test)
    y_pred_train=model.predict(df_train)
    
    return y_pred_test, y_pred_train


def lass(df_train, y_train, df_test):
    lasso= Lasso()
    param_grid = dict(max_iter = range(200,210),alpha=np.arange(0.1,1,.1))
    model_lasso=grip_search(df_train,y_train,lasso,param_grid)
    y_pred_train=model_lasso.predict(df_train)
    y_pred_test=model_lasso.predict(df_test)
    
    return y_pred_test, y_pred_train

def rg(df_train, y_train, df_test):
    rid=Ridge()
    param_grid = dict(alpha=np.arange(0.1,10,.5))
    model_ridge=grip_search(df_train,y_train,rid,param_grid)
    y_pred_train=model_ridge.predict(df_train)
    y_pred_test=model_ridge.predict(df_test)
    
    return y_pred_test, y_pred_train



def ElasN(df_train, y_train, df_test):
    elastic=ElasticNet(max_iter=10000)
    param_grid = dict(alpha=np.arange(0.1,10,.5))
    model_elastic=grip_search(df_train,y_train,elastic,param_grid)
    y_pred_train=model_elastic.predict(df_train)
    y_pred_test=model_elastic.predict(df_test)
    
    return y_pred_test, y_pred_train


def AdaB(df_train, y_train, df_test):
    ada=AdaBoostRegressor()
    param_grid={"n_estimators":range(50,200,10),"learning_rate":np.arange(0.1,2,.3)}
    model_ada=grip_search(df_train,y_train,ada,param_grid)
    y_pred_train=model_ada.predict(df_train)
    y_pred_test=model_ada.predict(df_test)
    
    return y_pred_test, y_pred_train


def svrm(df_train, y_train, df_test):
    svr=SVR()
    param_grid = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'],degree=range(2,4),epsilon=np.arange(0.1,1,.5))
    svr_m=grip_search(df_train,y_train,svr,param_grid)
    y_pred_train=svr_m.predict(df_train)
    y_pred_test=svr_m.predict(df_test)
    
    return y_pred_test, y_pred_train


def GradientB(df_train, y_train, df_test):
    model=GradientBoostingRegressor()
    param_grid = dict(loss=['ls', 'lad', 'huber'],
                 n_estimators=np.arange(50,200,10),
                  criterion=[ 'mse', 'mae'])
    gb=grip_search(df_train,y_train,model,param_grid)
    y_pred_train=gb.predict(df_train)
    y_pred_test=gb.predict(df_test)
    
    
    return y_pred_test, y_pred_train


def randomF(df_train, y_train, df_test):
    random= RandomForestRegressor()

    param_grid = dict(n_estimators = range(150,220,50),
                 criterion=["mse", "mae"],
                  bootstrap=[True,False],
                  oob_score=[True,False])
    model_random=grip_search(df_train,y_train,random,param_grid)
    y_pred_train=model_random.predict(df_train)
    y_pred_test=model_random.predict(df_test)
    
    return y_pred_test, y_pred_train


def KNN(df_train, y_train, df_test):
    knn=KNeighborsRegressor()
    param_knn = dict(n_neighbors = range(5,91))
    model_knn=grip_search(df_train,y_train,knn,param_knn)
    y_pred_train=model_knn.predict(df_train)
    y_pred_test=model_knn.predict(df_test)
    
    return y_pred_test, y_pred_train

def dtree(df_train, y_train, df_test):
    arbol = DecisionTreeRegressor()

    param_grid = dict(criterion = ["mse", "mae"],
                 max_depth=range(5,50,5))
    
    model_arbol=grip_search(df_train,y_train,arbol,param_grid)
    y_pred_train=model_arbol.predict(df_train)
    y_pred_test=model_arbol.predict(df_test)
    
    return y_pred_test, y_pred_train

def KRR(df_train, y_train, df_test):
    ker_ridge=KernelRidge(kernel='rbf', gamma=0.1)

    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)}
    model_ker_ridge=grip_search(df_train,y_train,ker_ridge,param_grid)
    y_pred_train=model_ker_ridge.predict(df_train)
    y_pred_test=model_ker_ridge.predict(df_test)
    
    return y_pred_test, y_pred_train

def BRR(df_train, y_train, df_test):
    baye=BayesianRidge(n_iter=10000)
    param_grid={"lambda_1":np.arange(0,1,.1),"alpha_1":np.arange(0,1,.1)}
    model_baye=grip_search(df_train,y_train,baye,param_grid)
    y_pred_train=model_baye.predict(df_train)
    y_pred_test=model_baye.predict(df_test)
    
    return y_pred_test, y_pred_train

from  scipy.stats  import  shapiro 
def shapiro_wilk(df,col):
    stat , p =shapiro(list(df[col].dropna().values) ) 
    alpha = 0.05
    if p > alpha:
        print( 'La muestra parece una Normal (No se rechaza H0)')
    else:
        print('La muestra no parece una Normal (Se rechaza H0)')

    

def Regresion(modelos, df_train, y_train, df_test, y_test):
    metrica=pd.DataFrame()
    estabilidad_train=pd.DataFrame()
    estabilidad_test=pd.DataFrame()
    estabilidad_train["tgt"]=y_train["tgt"]
    estabilidad_test["tgt"]=y_test["tgt"]
    lower_model = GradientBoostingRegressor(loss = "quantile", alpha = 0.1)  
    upper_model = GradientBoostingRegressor(loss = "quantile", alpha = 0.9)
    
    lower_model.fit(df_train, y_train) 
    upper_model.fit(df_train, y_train)
    estabilidad_train["lower_pred"]=lower_model.predict(df_train)
    estabilidad_train["upper_pred"]=upper_model.predict(df_train)
    estabilidad_test["lower_pred"]=lower_model.predict(df_test)
    estabilidad_test["upper_pred"]=upper_model.predict(df_test)
    
    for m in modelos:
        if m == 'Regresion Lineal':
            y_pred_test, y_pred_train=reglin(df_train, y_train, df_test)
            ms_1=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_1])
            estabilidad_train["pred_train_regL"]=y_pred_train
            estabilidad_train["res_train_regL"]=estabilidad_train[["tgt","pred_train_regL"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_regL"]=y_pred_test
            estabilidad_test["res_test_regL"]=estabilidad_test[["tgt","pred_test_regL"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            
        elif m == 'Lasso':
            y_pred_test, y_pred_train=lass(df_train, y_train, df_test)
            ms_2=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_2])
            estabilidad_train["pred_train_Lasso"]=y_pred_train
            estabilidad_train["res_train_Lasso"]=estabilidad_train[["tgt","pred_train_Lasso"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_Lasso"]=y_pred_test
            estabilidad_test["res_test_Lasso"]=estabilidad_test[["tgt","pred_test_Lasso"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'Ridge':
            y_pred_test, y_pred_train=rg(df_train, y_train, df_test)
            ms_3=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_3])
            estabilidad_train["pred_train_Ridge"]=y_pred_train
            estabilidad_train["res_train_Ridge"]=estabilidad_train[["tgt","pred_train_Ridge"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_Ridge"]=y_pred_test
            estabilidad_test["res_test_Ridge"]=estabilidad_test[["tgt","pred_test_Ridge"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'ElasticNet':
            y_pred_test, y_pred_train=ElasN(df_train, y_train, df_test)
            ms_4=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_4])
            estabilidad_train["pred_train_ElasticNet"]=y_pred_train
            estabilidad_train["res_train_ElasticNet"]=estabilidad_train[["tgt","pred_train_ElasticNet"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_ElasticNet"]=y_pred_test
            estabilidad_test["res_test_ElasticNet"]=estabilidad_test[["tgt","pred_test_ElasticNet"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'AdaBoost':
            y_pred_test, y_pred_train=AdaB(df_train, y_train, df_test)
            ms_5=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_5])
            estabilidad_train["pred_train_AdaBoost"]=y_pred_train
            estabilidad_train["res_train_AdaBoost"]=estabilidad_train[["tgt","pred_train_AdaBoost"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_AdaBoost"]=y_pred_test
            estabilidad_test["res_test_AdaBoost"]=estabilidad_test[["tgt","pred_test_AdaBoost"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'SVR':
            y_pred_test, y_pred_train=svrm(df_train, y_train, df_test)
            ms_6=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_6])
            estabilidad_train["pred_train_SVR"]=y_pred_train
            estabilidad_train["res_train_SVR"]=estabilidad_train[["tgt","pred_train_SVR"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_SVR"]=y_pred_test
            estabilidad_test["res_test_SVR"]=estabilidad_test[["tgt","pred_test_SVR"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'Gradient Boost':
            y_pred_test, y_pred_train=GradientB(df_train, y_train, df_test)
            ms_7=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_7])
            estabilidad_train["pred_train_GB"]=y_pred_train
            estabilidad_train["res_train_GB"]=estabilidad_train[["tgt","pred_train_GB"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_GB"]=y_pred_test
            estabilidad_test["res_test_GB"]=estabilidad_test[["tgt","pred_test_GB"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'Random Forest':
            y_pred_test, y_pred_train=randomF(df_train, y_train, df_test)
            ms_8=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_8])
            estabilidad_train["pred_train_RF"]=y_pred_train
            estabilidad_train["res_train_RF"]=estabilidad_train[["tgt","pred_train_RF"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_RF"]=y_pred_test
            estabilidad_test["res_test_RF"]=estabilidad_test[["tgt","pred_test_RF"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'KNN':
            y_pred_test, y_pred_train=KNN(df_train, y_train, df_test)
            ms_9=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_9])
            estabilidad_train["pred_train_KNN"]=y_pred_train
            estabilidad_train["res_train_KNN"]=estabilidad_train[["tgt","pred_train_KNN"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_KNN"]=y_pred_test
            estabilidad_test["res_test_KNN"]=estabilidad_test[["tgt","pred_test_KNN"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'Desicion Tree':
            y_pred_test, y_pred_train=dtree(df_train, y_train, df_test)
            ms_10=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_10])
            estabilidad_train["pred_train_tree"]=y_pred_train
            estabilidad_train["res_train_tree"]=estabilidad_train[["tgt","pred_train_tree"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_tree"]=y_pred_test
            estabilidad_test["res_test_tree"]=estabilidad_test[["tgt","pred_test_tree"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'Kernel Ridge Regression':
            y_pred_test, y_pred_train=KRR(df_train, y_train, df_test)
            ms_11=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_11])
            estabilidad_train["pred_train_KRR"]=y_pred_train
            estabilidad_train["res_train_KRR"]=estabilidad_train[["tgt","pred_train_KRR"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_KRR"]=y_pred_test
            estabilidad_test["res_test_KRR"]=estabilidad_test[["tgt","pred_test_KRR"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
        elif m == 'Bayesian Ridge Regression':
            y_pred_test, y_pred_train=BRR(df_train, y_train, df_test)
            ms_12=metricas(m,df_train, df_test, y_train, y_pred_train, y_test, y_pred_test)
            metrica=pd.concat([metrica, ms_12])
            estabilidad_train["pred_train_BRR"]=y_pred_train
            estabilidad_train["res_train_BRR"]=estabilidad_train[["tgt","pred_train_BRR"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
            estabilidad_test["pred_test_BRR"]=y_pred_test
            estabilidad_test["res_test_BRR"]=estabilidad_test[["tgt","pred_test_BRR"]].apply(lambda x:abs(x[0]-x[1]),axis=1)
    
    metrica.reset_index(drop=True, inplace=True)
    estabilidad_train.reset_index(drop=True, inplace=True)
    estabilidad_test.reset_index(drop=True, inplace=True)
    
    
    return metrica, estabilidad_train, estabilidad_test

def limpieza(df):
    
    df.drop('key', axis=1,inplace= True)

    c_feats=['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude', 'dropoff_latitude']
    v_feats=['passenger_count', 'fare_class']
    d_feats=['pickup_datetime'] 

    c_feats_new=["c_"+x for x in c_feats]
    v_feats_new=["v_"+x for x in v_feats]
    d_feats_new=["d_"+x for x in d_feats]

    df.rename(columns=dict(zip(d_feats,d_feats_new)),inplace=True)
    df.rename(columns=dict(zip(v_feats,v_feats_new)),inplace=True)
    df.rename(columns=dict(zip(c_feats,c_feats_new)),inplace=True)
    
    df.dropna(inplace= True)
    
    df['d_pickup_datetime']=pd.to_datetime(df['d_pickup_datetime'])
    
    df['d_weekday']=df['d_pickup_datetime'].map(lambda x: x.isoweekday())
    df['d_day']=df['d_pickup_datetime'].map(lambda x: x.day)
    df['d_month']=df['d_pickup_datetime'].map(lambda x: x.month)
    df['d_year']=df['d_pickup_datetime'].map(lambda x: x.year)
    df['d_hour']=df['d_pickup_datetime'].map(lambda x: x.hour)
    df['d_minute']=df['d_pickup_datetime'].map(lambda x: x.minute)
    df['d_yearday']=df['d_pickup_datetime'].map(lambda x: x.timetuple().tm_yday)
    df['v_quarter']=df['d_pickup_datetime'].dt.quarter
    
    
    
    df=df[~((df.c_pickup_longitude == 0)&(df.c_dropoff_longitude==0)&(df.c_pickup_latitude== 0)&(df.c_dropoff_latitude==0))]
    df=df[~(df.c_pickup_longitude == 0)]
    df=df[~(df.c_pickup_latitude == 0)]
    df=df[~(df.c_dropoff_longitude == 0)]
    df=df[~(df.c_pickup_latitude==0)]
    
    
    df=df[~((df.c_pickup_longitude == df.c_dropoff_longitude)&(df.c_pickup_latitude == df.c_dropoff_latitude))]
    
    df['c_km']=km(df,'c_pickup_latitude', 'c_pickup_longitude', 'c_dropoff_latitude', 'c_dropoff_longitude')
    
    cols=['c_fare_amount', 'c_pickup_longitude','c_pickup_latitude', 'c_dropoff_longitude', 'c_dropoff_latitude',
      'c_km']
    outl=OUTLIERS(df, cols)
    
    df=dropoutliers(df,outl)
    
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    y=df['c_fare_amount']
    y_class=df['v_fare_class']
    df=df.drop(['d_pickup_datetime','v_fare_class','c_fare_amount'],axis=1)
    

    X=df[[x for x in df.columns if x != tgt]]
    

    return  X, y, y_class

def grip_search_c(X_train,y_train,estimator,param_grid):
    grid = GridSearchCV(
                  verbose=True,
                  scoring='roc_auc',
                  estimator=estimator,
                  n_jobs=-1,
                  param_grid=param_grid)
    grid.fit(X_train,y_train)
    print(f"Best Score : {grid.best_score_}")
    print(f"Best Params : {grid.best_params_}")
    return grid.best_estimator_


def metrics(cm, y, y_score, modelo,datos):
    vn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    vp=cm[1][1]
    
    
    fpr, tpr, thresholds = roc_curve(y, y_score)

    
    accuracy=(vp+vn)/(vp+fp+fn+vn)

    
    precision=vp/(vp+fp)


    recall=(vp/(vp+fn))


    f1_score=((2*precision*recall)/(recall+precision))

    
    TPR=recall


    FPR=(fp/(fp+vn))
    
    ms=pd.DataFrame(columns=('modelo','datos','ROC','Exactitud','Precision','Recall','F1_score','TPR','FPR'))
    ms.loc[len(ms)]=list([modelo,datos,round(auc(fpr, tpr)*100,2),round(accuracy*100,2),round(precision*100,2),round(recall*100,2),round(f1_score*100,2),round(TPR*100,2),round(FPR*100,2)])
    
    return ms

def KNN_c(df_train, y_train, df_test):
    knn=KNeighborsClassifier()
    param_knn = dict(n_neighbors = range(5,91))
    model_knn=grip_search_c(df_train,y_train,knn,param_knn)
    y_pred_train=model_knn.predict(df_train)
    y_pred_test=model_knn.predict(df_test)
    y_score_train=model_knn.predict_proba(df_train)[:,1]
    y_score_test=model_knn.predict_proba(df_test)[:,1]
    
    return y_pred_test, y_pred_train, y_score_test, y_score_train, model_knn

def svm(df_train, y_train, df_test):
    classifier_ker = SVC(random_state = 0,probability=True)
    param_grid = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'],degree=[2,3,4])
    svr_m=grip_search_c(df_train,y_train,svr,param_grid)
    y_pred_train=svr_m.predict(df_train)
    y_pred_test=svr_m.predict(df_test)
    y_score_train=svr_m.predict_proba(df_train)[:,1]
    y_score_test=svr_m.predict_proba(df_test)[:,1]
    
    return y_pred_test, y_pred_train, y_score_test, y_score_train, svr_m


def dtree_c(df_train, y_train, df_test):
    arbol = DecisionTreeClassifier()
    param_grid = dict(criterion = ["gini", "entropy"],
                 max_depth=[6,7,8])
    model_arbol=grip_search_c(df_train,y_train,arbol,param_grid)
    y_pred_train=model_arbol.predict(df_train)
    y_pred_test=model_arbol.predict(df_test)
    y_score_train=model_arbol.predict_proba(df_train)[:,1]
    y_score_test=model_arbol.predict_proba(df_test)[:,1]
    
    return y_pred_test, y_pred_train, y_score_test, y_score_train, model_arbol

def gradient(df_train, y_train, df_test):
    sgd = SGDClassifier()

    param_grid = dict(
                 penalty=["l2","l1","elasticnet"],
                 alpha=np.arange(0.1,.3,.01))
    model_sgd=grip_search_c(df_train,y_train,sgd,param_grid)
    y_pred_train=model_sgd.predict(df_train)
    y_pred_test=model_sgd.predict(df_test)
    y_score_train=model_sgd.predict_proba(df_train)[:,1]
    y_score_test=model_sgd.predict_proba(df_test)[:,1]
    
    return y_pred_test, y_pred_train, y_score_test, y_score_train, model_sgd

def logireg(df_train, y_train, df_test):
    log= LogisticRegression()

    param_grid = dict(solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                 penalty=["l2","l1","elasticnet"])
    model_log=grip_search_c(df_train,y_train,log,param_grid)
    y_pred_train=model_log.predict(df_train)
    y_pred_test=model_log.predict(df_test)
    y_score_train=model_log.predict_proba(df_train)[:,1]
    y_score_test=model_log.predict_proba(df_test)[:,1]
    
    return y_pred_test, y_pred_train, y_score_test, y_score_train, model_log

def adaboost_c(df_train, y_train, df_test):
    ada= AdaBoostClassifier()

    param_grid = dict(learning_rate=np.arange(0.1,1,.1))
    model_ada=grip_search_c(df_train,y_train,ada,param_grid)
    y_pred_train=model_ada.predict(df_train)
    y_pred_test=model_ada.predict(df_test)
    y_score_train=model_ada.predict_proba(df_train)[:,1]
    y_score_test=model_ada.predict_proba(df_test)[:,1]
    
    return y_pred_test, y_pred_train, y_score_test, y_score_train, model_ada

def resultados(df,model,tgt):
        
    df_re=df.copy()
    df_re["pred"]=model.predict(df[[x for x in df.columns if x != tgt]])
    df_re["proba"]=model.predict_proba(df[[x for x in df.columns if x != tgt]])[:,1]
    proba=[]
    n_fraude=[]
    fraude=[]
    for i in [pd.Interval(-1,.1),pd.Interval(.1,.2),pd.Interval(.2,.3),pd.Interval(.3,.4),pd.Interval(.4,.5),pd.Interval(.5,.6),pd.Interval(.6,.7),pd.Interval(.7,.8),pd.Interval(.8,.9),pd.Interval(.9,1)]:
        aux={0:0,1:0}
        dictio=df_re[df_re["proba"].map(lambda x:x in i)][tgt].value_counts().to_dict()
        aux.update(dictio)
        n_fraude.append(aux[0])
        fraude.append(aux[1])
        proba.append("("+str(round(i.left,1))+","+str(round(i.right,1))+"]")
    resultado=pd.DataFrame({"Proba":proba,"low_fare":n_fraude,"high_fare":fraude})
    
    return resultado




def Classif(modelos, df_train, y_train, df_test, y_test, data):
    metrica=pd.DataFrame()
    estabilidad=pd.DataFrame()

    for m in modelos:
        if m == 'Regresion Logistica':
            y_pred_test, y_pred_train, y_score_test, y_score_train, model_log=logireg(df_train, y_train, df_test)
            ms_1=metrics(confusion_matrix(y_train, y_pred_train), y_train, y_score_train, m, 'train')
            mts_1=metrics(confusion_matrix(y_test, y_pred_test), y_test, y_score_test, m, 'test')
            metrica=pd.concat([metrica, ms_1])
            metrica=pd.concat([metrica, mts_1])
            resultado=resultados(data, model_log, 'tgt')
            estabilidad=pd.concat([estabilidad, resultado])

        elif m == 'AdaBoost':
            y_pred_test, y_pred_train, y_score_test, y_score_train, model_ada=adaboost_c(df_train, y_train, df_test)
            ms_5=metrics(confusion_matrix(y_train, y_pred_train), y_train, y_score_train, m, 'train')
            mts_5=metrics(confusion_matrix(y_test, y_pred_test), y_test, y_score_test, m, 'test')
            metrica=pd.concat([metrica, ms_5])
            metrica=pd.concat([metrica, mts_5])
            resultado=resultados(data, model_ada, 'tgt')
            estabilidad=pd.concat([estabilidad, resultado])
            
        elif m == 'SVM':
            y_pred_test, y_pred_train, y_score_test, y_score_train, svr_m=svm(df_train, y_train, df_test)
            ms_6=metrics(confusion_matrix(y_train, y_pred_train), y_train, y_score_train, m, 'train')
            mts_6=metrics(confusion_matrix(y_test, y_pred_test), y_test, y_score_test, m, 'test')
            metrica=pd.concat([metrica, ms_6])
            metrica=pd.concat([metrica, mts_6])
        elif m == 'Gradient D':
            y_pred_test, y_pred_train, y_score_test, y_score_train, model_sgd=gradient(df_train, y_train, df_test)
            ms_7=metrics(confusion_matrix(y_train, y_pred_train), y_train, y_score_train, m, 'train')
            mts_7=metrics(confusion_matrix(y_test, y_pred_test), y_test, y_score_test, m, 'test')
            metrica=pd.concat([metrica, ms_7])
            metrica=pd.concat([metrica, mts_7])
            resultado=resultados(data, model_sgd, 'tgt')
            estabilidad=pd.concat([estabilidad, resultado])
        elif m == 'KNN':
            y_pred_test, y_pred_train, y_score_test, y_score_train, model_knn=KNN_c(df_train, y_train, df_test)
            ms_9=metrics(confusion_matrix(y_train, y_pred_train), y_train, y_score_train, m, 'train')
            mts_9=metrics(confusion_matrix(y_test, y_pred_test), y_test, y_score_test, m, 'test')
            metrica=pd.concat([metrica, ms_9])
            metrica=pd.concat([metrica, mts_9])
        elif m == 'Desicion Tree':
            y_pred_test, y_pred_train, y_score_test, y_score_train, model_arbol=dtree_c(df_train, y_train, df_test)
            ms_10=metrics(confusion_matrix(y_train, y_pred_train), y_train, y_score_train, m, 'train')
            mts_10=metrics(confusion_matrix(y_test, y_pred_test), y_test, y_score_test, m, 'test')
            metrica=pd.concat([metrica, ms_10])
            metrica=pd.concat([metrica, mts_10])
            resultado=resultados(data, model_arbol, 'tgt')
            estabilidad=pd.concat([estabilidad, resultado])
            
    metrica.reset_index(drop=True, inplace=True)
    
    
    return metrica, estabilidad

def prueba(df, data):
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.rename(columns={'tgt_price':'tgt'}, inplace=True)
    X=df[[x for x in df if x!="tgt"]]
    y=df["tgt"]
    
    file=data
    loaded_model=pickle.load(open(file,'rb'))
    result_reg=loaded_model.score(X,y)
    
    return X, y, loaded_model, result_reg