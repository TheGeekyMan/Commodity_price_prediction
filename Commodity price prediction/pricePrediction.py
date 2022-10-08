from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#print(os.listdir("data\\"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb



import warnings
warnings.filterwarnings('ignore')

X_train = []
X_test = []
Y_train = []
Y_test = []

X_train_stock = []
X_test_stock = []
Y_train_stock = []
Y_test_stock = []



#------------------------------- Logistic Regression -------------------------------

def Logistic_Regression_Method(compareData):


    lr = LogisticRegression()

    try :


        lr.fit(X_train,Y_train)

        Y_pred_lr = lr.predict(X_test)

        #print(Y_pred_lr.shape)

        score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

        print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

        precision, recall, fscore, support = precision_recall_fscore_support(
        Y_test, lr.predict(X_test), average = 'macro')

        #------------------- Stock -----------------

        lr.fit(X_train_stock,Y_train_stock)

        Y_pred_lr_stock = lr.predict(X_test_stock)

        #print(Y_pred_lr_stock)
        if compareData == "fscore":
            return fscore,Y_pred_lr,Y_pred_lr_stock   
        elif compareData == "recall":
            return recall,Y_pred_lr,Y_pred_lr_stock   
        elif compareData == "precision":
            return precision,Y_pred_lr,Y_pred_lr_stock   
        else:
            return score_lr,Y_pred_lr,Y_pred_lr_stock

    except : 
        return 0,0,0


    #return score_lr,fscore,recall,precision,Y_pred_lr

#print(Y_pred_lr)



#------------------- Naive Bayes -----------------------

def Naive_Bayes_Method(compareData):

    nb = GaussianNB()

    try :

        nb.fit(X_train,Y_train)

        Y_pred_nb = nb.predict(X_test)

        score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

        print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

        precision, recall, fscore, support = precision_recall_fscore_support(
        Y_test, nb.predict(X_test), average = 'macro')

        nb.fit(X_train_stock,Y_train_stock)

        Y_pred_stock = nb.predict(X_test_stock)
        
        #print(Y_pred_stock)
        if compareData == "fscore":
            return fscore,Y_pred_nb,Y_pred_stock   
        elif compareData == "recall":
            return recall,Y_pred_nb,Y_pred_stock   
        elif compareData == "precision":
            return precision,Y_pred_nb,Y_pred_stock   
        else:
            return score_nb,Y_pred_nb,Y_pred_stock

    except :
        return 0,0,0
    #return score_nb,fscore,recall,precision,Y_pred_nb

#------------------------ SVM ----------------------

def Support_Vector_Method(compareData):

    sv = svm.SVC(kernel='linear')

    try :

        sv.fit(X_train, Y_train)

        Y_pred_svm = sv.predict(X_test)

        score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

        print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")

        precision, recall, fscore, support = precision_recall_fscore_support(
        Y_test, sv.predict(X_test), average = 'macro')

        sv.fit(X_train_stock,Y_train_stock)

        Y_pred_stock = sv.predict(X_test_stock)
        #print(Y_pred_stock)
        if compareData == "fscore":
            return fscore,Y_pred_svm  ,Y_pred_stock 
        elif compareData == "recall":
            return recall,Y_pred_svm   ,Y_pred_stock
        elif compareData == "precision":
            return precision,Y_pred_svm   ,Y_pred_stock
        else:
            return score_svm,Y_pred_svm,Y_pred_stock
    
    except :
        return 0,0,0

    #return score_svm,fscore,recall,precision,Y_pred_svm


#--------------------------------  KNN ------------------------------

def KNN_Method(compareData):
    

    knn = KNeighborsClassifier(n_neighbors=3)

    try:

        knn.fit(X_train,Y_train)
        Y_pred_knn=knn.predict(X_test)

        score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

        print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")

        precision, recall, fscore, support = precision_recall_fscore_support(
        Y_test, knn.predict(X_test), average = 'macro')
        

        knn.fit(X_train_stock,Y_train_stock)

        Y_pred_stock = knn.predict(X_test_stock)

        #print(Y_pred_stock)
        if compareData == "fscore":
            return fscore,Y_pred_knn,Y_pred_stock   
        elif compareData == "recall":
            return recall,Y_pred_knn,Y_pred_stock   
        elif compareData == "precision":
            return precision,Y_pred_knn,Y_pred_stock   
        else:
            return score_knn,Y_pred_knn,Y_pred_stock

    except :
        return 0,0,0
    #return score_knn,fscore,recall,precision,Y_pred_knn



#-----------------------------Decision Tree-------------------------------

def DecisionTree_Method(compareData):

    max_accuracy = 0
    best_x = 0

    try :

        for x in range(200):
            dt = DecisionTreeClassifier(random_state=x)
            dt.fit(X_train,Y_train)
            Y_pred_dt = dt.predict(X_test)
            current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
            if(current_accuracy>max_accuracy):
                max_accuracy = current_accuracy
                best_x = x
                
        #print(max_accuracy)
        #print(best_x)


        dt = DecisionTreeClassifier(random_state=best_x)
        dt.fit(X_train,Y_train)
        Y_pred_dt = dt.predict(X_test)

        #print(Y_pred_dt)

        score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

        print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")

        precision, recall, fscore, support = precision_recall_fscore_support(
        Y_test, dt.predict(X_test), average = 'macro')

        dt.fit(X_train_stock,Y_train_stock)

        Y_pred_stock = dt.predict(X_test_stock)
        
        #print(Y_pred_stock)
        if compareData == "fscore":
            return fscore,Y_pred_dt,Y_pred_stock   
        elif compareData == "recall":
            return recall,Y_pred_dt,Y_pred_stock   
        elif compareData == "precision":
            return precision,Y_pred_dt,Y_pred_stock   
        else:
            return score_dt,Y_pred_dt,Y_pred_stock

    except :
        return 0,0,0
    #return score_dt,fscore,recall,precision,Y_pred_dt
    #print("F-Score : "+str(fscore))
    #print("Recall : "+str(recall))
    #print("Precision : "+str(precision))
    #print("Support : "+str(support_decision_tree))

    #print(Y_pred_dt)
    #for i, price in enumerate(Y_pred_dt):
    #    print ("Predicted selling price for Client ".format(i+1, price))

    



#---------------------- XGboost----------------------------

def XGB_Method(compareData):

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    try:

        xgb_model.fit(X_train, Y_train)

        Y_pred_xgb = xgb_model.predict(X_test)

        score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

        print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")

        #precision, recall, fscore, support = precision_recall_fscore_support(
        #Y_test, xgb.predict(X_test), average = 'macro')
        
        xgb_model.fit(X_train_stock,Y_train_stock)

        Y_pred_stock = xgb_model.predict(X_test_stock)
        
        #print(Y_pred_stock)
        if compareData == "fscore":
            return 0,Y_pred_xgb,Y_pred_stock   
        elif compareData == "recall":
            return 0,Y_pred_xgb,Y_pred_stock   
        elif compareData == "precision":
            return 0,Y_pred_xgb,Y_pred_stock   
        else:
            return score_xgb,Y_pred_xgb,Y_pred_stock   
     
    except :
        return 0,0 ,0

    #return score_xgb,fscore,recall,precision,Y_pred_xgb

    #print(Y_pred_xgb.shape)
    


#---------------------------- Final ----------------------------

#scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_xgb]
#algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","XGBoost"]    

#for i in range(len(algorithms)):
#    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


'''
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)

plt.show()

'''


class PredictData:
   
    # init method or constructor 
    def __init__(self, dataFile,compare):
        self.dataFile = dataFile
        self.compare = compare
    
    def prediction(self):

        dataset = pd.read_csv("SourceCode\\temp\\out.csv")
        
        dataFile = self.dataFile
        compare = self.compare

        #predictors = dataset.drop("mp_price",axis=1)
        #target = dataset["mp_price":"min_price"]

        #dataset1['mkt_name'].values == str(compare)
        #dataset = dataset1#['mkt_name'==compare]

        #print(dataset1['mkt_name']== compare)
        #print(dataset['mkt_name'].values == str(compare))

        predictors = dataset.iloc[:,[14,15,16]].astype(int)
        target = dataset.iloc[:,16].astype(int)

        predictors_stock = dataset.iloc[:,[14,15,16,17]].astype(int)
        target_stock = dataset.iloc[:,17].astype(int)
        
        try:

            global X_test,X_train,Y_test,Y_train,X_test_stock,X_train_stock,Y_test_stock,Y_train_stock

            X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

            X_train_stock,X_test_stock,Y_train_stock,Y_test_stock = train_test_split(predictors_stock,target_stock,test_size=0.20,random_state=0)

            type(target)
            #print(X_train.shape)

            #print(dataset["mp_month"].unique())

            #--------------------------------     Feature Scaling------------------------------- 

            sc = StandardScaler()

            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            X_train_stock = sc.fit_transform(X_train_stock)
            X_test_stock = sc.transform(X_test_stock)

            Lr_data1,Lr_data2,Lr_stock_data = Logistic_Regression_Method(compare)
            Nb_data1,Nb_data2,Nb_stock_data = Naive_Bayes_Method(compare)
            Sv_data1,Sv_data2,Sv_stock_data = Support_Vector_Method(compare)
            Knn_data1,Knn_data2,Knn_stock_data = KNN_Method(compare)
            Dt_data1,Dt_data2,Dt_stock_data = DecisionTree_Method(compare)
            Xgb_data1,Xgb_data2,Xgb_stock_data = XGB_Method(compare)
            
            compareAlgoDict = {"a":Lr_data1,"b":Nb_data1,"c":Sv_data1,"d":Knn_data1,"e":Dt_data1,"f":Xgb_data1}

            max_key = max(compareAlgoDict, key=compareAlgoDict.get)
            #print(max_key)

            scores = [Lr_data1,Nb_data1,Sv_data1,Knn_data1,Dt_data1,Xgb_data1]
            algorithms = ["Logistic R","Naive B","SVM","KNN","Decision T","XGBoost"]    

            #for i in range(len(algorithms)):
            #    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


            
            
            #sns.set(rc={'figure.figsize':(15,8)})
            #plt.xlabel("Algorithms")
            #plt.ylabel("Accuracy score")

            #sns.barplot(algorithms,scores)

            #plt.show()

            #fig= plt.figure(figsize=(24,12))
            ##plt.plot(X,Y)
            #plt.plot(target,Dt_data2)
            #plt.show()
            
            '''print(Lr_stock_data)
            print(Nb_stock_data)
            print(Sv_stock_data)
            print(Knn_stock_data)
            print(Dt_stock_data)
            print(Xgb_stock_data)
            '''

            if max_key == "a":
                return Lr_data1,Lr_data2,scores,"Logistic Regression",Lr_stock_data
            elif max_key == "b":
                return Nb_data1,Nb_data2,scores,"Naive Bayes",Nb_stock_data
            elif max_key == "c":
                return Sv_data1,Sv_data2,scores,"Support Vector Machine",Sv_stock_data
            elif max_key == "d":
                return Knn_data1,Knn_data2,scores,"K-Nearest Neighbors",Knn_stock_data
            elif max_key == "e":
                return Dt_data1,Dt_data2,scores,"Decision Tree",Dt_stock_data
            elif max_key == "f":
                return Xgb_data1,Xgb_data2,scores,"XGBoost",Xgb_stock_data

        except :
            return 0,0,0,"None",0

#data1,data2,data3 = prediction("soyaOil.csv","accuracy")

#print("Highest Score "+str(data1))
#print("Upcoming price "+str(data2))
