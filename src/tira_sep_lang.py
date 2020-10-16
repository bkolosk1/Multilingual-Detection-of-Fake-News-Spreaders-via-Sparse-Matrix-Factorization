# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:42:30 2020
@author: Bosec
"""


## some more experiments
import xml.etree.ElementTree as ET
import config 
import numpy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
import parse_data
import time
import csv
import config 
from feature_construction import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import pickle

try:
    import umap
except:
    pass

def train(X,Y,output=False):
    final_y = []
    final_texts = []
    
    for k,v in Y.items():
        text = X[k]
        label = v
        final_texts.append(text)
        final_y.append(v)

    dataframe = build_dataframe(final_texts)

    report = []

    trained_models = {}
    
    for nrep in range(5):
        for nfeat in [500,2500,5000,10000,15000]:
            for dim in [256,512,768]:
                tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = nfeat, labels = final_y)

                reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1))
 #               reducer = umap.UMAP(n_components = min(dim, nfeat * len(feature_names)-1))
                data_matrix = reducer.fit_transform(data_matrix)

                X_train, X_test, y_train, y_test = train_test_split(data_matrix, final_y, train_size=0.9, test_size=0.1)

                logging.info("Generated {} features.".format(nfeat*len(feature_names)))
           #     parameters = {'kernel':["linear","poly"], 'C':[0.1, 1, 10, 100, 500],"gamma":["scale","auto"],"class_weight":["balanced",None]}
                parameters = {"loss":["hinge","log"],"penalty":["elasticnet"],"alpha":[0.01,0.001,0.0001,0.0005],"l1_ratio":[0.05,0.25,0.3,0.6,0.8,0.95],"power_t":[0.5,0.1,0.9]}
#                svc = svm.SVC()
                svc = SGDClassifier()
                clf1 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
                scores = cross_val_score(clf1, data_matrix, final_y, cv=5)
                acc_svm = scores.mean()
                logging.info("SGD 5fCV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

                parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l2"]}
                svc = LogisticRegression(max_iter = 100000)
                clf2 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = 8,cv = 10, refit = True)
                scores = cross_val_score(clf2, data_matrix, final_y, cv=5)
                logging.info("LR 5fCV ccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
                acc_lr = scores.mean()
                trained_models[nfeat] = ((clf1, clf2), dim)
                report.append([nfeat, acc_lr, acc_svm])
    
    dfx = pd.DataFrame(report)
    dfx.columns = ["Number of features","LR","SVM"]
    dfx = pd.melt(dfx, id_vars=['Number of features'], value_vars=['LR','SVM'])
    sns.lineplot(dfx['Number of features'],dfx['value'], hue = dfx["variable"], markers = True, style = dfx['variable'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.PATH_IMGS+"/tfidif-expanded-5fCV.png",dpi = 300)
    sorted_dfx = dfx.sort_values(by = ["value"])
    print(sorted_dfx.iloc[-1,:])
    max_acc = sorted_dfx.iloc[-1,:][['Number of features','variable']]

    final_feature_number = max_acc['Number of features']
    final_learner = max_acc['variable']
    logging.info(" Final feature number: {}, final learner: {}".format(final_feature_number, final_learner))
    
    if final_learner == "SVM":
        index = 0        
    else:
        index = 1

    clf_final, dim = trained_models[final_feature_number]
    clf_final = clf_final[index]
    tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = final_feature_number)
    reducer = TruncatedSVD(n_components = min(dim, nfeat * len(feature_names)-1)).fit(data_matrix)
    return tokenizer, clf_final, reducer

def export():
    XRaw,YRaw = parse_data.export(config.PATH_DATA_EN,config.PATH_DATA_EN_TRUTH)
    tokenizer, clf, reducer = train(XRaw, YRaw)
    with open(os.path.join(config.PATH_MODELS,"tokenizer_5fcv_en.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(config.PATH_MODELS,"clf_5fcv_en.pkl"),mode='wb') as f:
        pickle.dump(clf,f)
    with open(os.path.join(config.PATH_MODELS,"reducer_5fcv_en.pkl"),mode='wb') as f:
        pickle.dump(reducer,f)
    #ES
    XRaw,YRaw = parse_data.export(config.PATH_DATA_ES,config.PATH_DATA_ES_TRUTH)
    tokenizer, clf, reducer = train(XRaw, YRaw)
    with open(os.path.join(config.PATH_MODELS,"tokenizer_5fcv_es.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(config.PATH_MODELS,"clf_5fcv_es.pkl"),mode='wb') as f:
        pickle.dump(clf,f)
    with open(os.path.join(config.PATH_MODELS,"reducer_5fcv_es.pkl"),mode='wb') as f:
        pickle.dump(reducer,f)

def _import(lang,path_in=config.PATH_MODELS):
    """Imports tokenizer,clf,reducer from param(path_in, default is ../models)"""
    tokenizer = pickle.load(open(os.path.join(path_in,"tokenizer_5fcv_"+lang+".pkl"),'rb'))
    clf = pickle.load(open(os.path.join(path_in,"clf_5fcv_"+lang+".pkl"),'rb'))
    reducer = pickle.load(open(os.path.join(path_in,"reducer_5fcv"+lang+".pkl"),'rb'))
    return tokenizer,clf,reducer

def fit(path,out_path=config.PATH_OUT,lang='en'):
    """Fits data from param(path), outputs xml file as out_path"""
    #print("TUKA")
    tokenizer,clf,reducer = _import(lang)
    #print("DATA IMPORTED")
    #XRaw,YRaw = parse_data.export()
    test_texts,name_idx = parse_data.exportTest(path)
    #print("TESTS IMPORTED")
    df_text = build_dataframe(test_texts)
    matrix_form = tokenizer.transform(df_text)
    reduced_matrix_form = reducer.transform(matrix_form)
    predictions = clf.predict(reduced_matrix_form)
    for i in range(len(name_idx)):
        out_name = name_idx[i]+".xml"
        root = ET.Element("author")
        root.set('id',name_idx[i])
        root.set('lang',lang)
        root.set('type',str(predictions[i]))
        tree = ET.ElementTree(root)
        tree.write(os.path.join(out_path,out_name))
            
if __name__ == "__main__":
    export()
    #tokenizer, clf, reducer = train(XRaw, YRaw)
    #export(tokenizer, clf, reducer)
    #tokenizer,clf,reducer = _import()
