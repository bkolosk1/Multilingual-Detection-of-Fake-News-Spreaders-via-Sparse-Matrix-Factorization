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
from sklearn.linear_model import LogisticRegression
import parse_data
import time
import csv
import config 
from feature_construction import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

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
    for nfeat in [500,1000,5000,10000,20000]:

        tokenizer, feature_names, data_matrix = get_features(dataframe, max_num_feat = nfeat)
        X_train, X_test, y_train, y_test = train_test_split(data_matrix, final_y, train_size=0.9, test_size=0.1)

        logging.info("Generated {} features.".format(nfeat*len(feature_names)))
        parameters = {'kernel':["linear","poly"], 'C':[0.1, 1, 10, 100, 500],"gamma":["scale","auto"],"class_weight":["balanced",None]}
        svc = svm.SVC()
        clf1 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = 8)
        clf1.fit(X_train, y_train)
        logging.info(str(max(clf1.cv_results_['mean_test_score'])) +" training configuration with best score (SVM)")

        predictions = clf1.predict(X_test)
        acc_svm = accuracy_score(predictions,y_test)
        logging.info("Test accuracy score SVM {}".format(acc_svm))

        parameters = {"C":[0.1,1,10,25,50,100,500],"penalty":["l2"]}
        svc = LogisticRegression(max_iter = 100000)
        clf2 = GridSearchCV(svc, parameters, verbose = 0, n_jobs = 8)
        clf2.fit(X_train, y_train)
        logging.info(str(max(clf2.cv_results_['mean_test_score'])) + " training configuration with best score (LR)")

        predictions = clf2.predict(X_test)
        acc_lr = accuracy_score(predictions,y_test)
        logging.info("Test accuracy score SVM {}".format(acc_lr))
        trained_models[nfeat] = (clf1, clf2)
        report.append([nfeat, acc_lr, acc_svm])
    
    dfx = pd.DataFrame(report)
    dfx.columns = ["Number of features","LR","SVM"]
    dfx = pd.melt(dfx, id_vars=['Number of features'], value_vars=['LR','SVM'])
    sns.lineplot(dfx['Number of features'],dfx['value'], hue = dfx["variable"], markers = True, style = dfx['variable'])
    plt.legend()
    plt.tight_layout()
    plt.savefig("report.png",dpi = 300)
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

    clf_final = trained_models[final_feature_number][index]    
    return tokenizer, clf_final

def export():
    XRaw,YRaw = parse_data.export(config.PATH_DATA_EN,config.PATH_DATA_EN_TRUTH)
    tokenizer, clf, reducer = train(XRaw, YRaw)
    with open(os.path.join(config.PATH_MODELS,"tokenizer_en.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(config.PATH_MODELS,"clf_en.pkl"),mode='wb') as f:
        pickle.dump(clf,f)
    with open(os.path.join(config.PATH_MODELS,"reducer_en.pkl"),mode='wb') as f:
        pickle.dump(reducer,f)
    #ES
    XRaw,YRaw = parse_data.export(config.PATH_DATA_ES,config.PATH_DATA_ES_TRUTH)
    tokenizer, clf, reducer = train(XRaw, YRaw)
    with open(os.path.join(config.PATH_MODELS,"tokenizer_es.pkl"),mode='wb') as f:
        pickle.dump(tokenizer,f)
    with open(os.path.join(config.PATH_MODELS,"clf_es.pkl"),mode='wb') as f:
        pickle.dump(clf,f)
    with open(os.path.join(config.PATH_MODELS,"reducer_es.pkl"),mode='wb') as f:
        pickle.dump(reducer,f)

def _import(lang,path_in=config.PATH_MODELS):
    """Imports tokenizer,clf,reducer from param(path_in, default is ../models)"""
    tokenizer = pickle.load(open(os.path.join(path_in,"tokenizer_"+lang+".pkl"),'rb'))
    clf = pickle.load(open(os.path.join(path_in,"clf_"+lang+".pkl"),'rb'))
    reducer = pickle.load(open(os.path.join(path_in,"reducer_"+lang+".pkl"),'rb'))
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
