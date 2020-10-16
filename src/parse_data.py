import xml.etree.ElementTree as ET
import config 
import numpy
import os
import csv
from sklearn.model_selection import train_test_split

def readListDir(path):
    "Returns list of articles present in the FS."
    return os.listdir(path)


def readLabels(path=config.PATH_DATA_EN_TRUTH):
    """Returns labelsof each document"""
    labels = {}
    with open(path,"r") as file:
        for line in file: 
            line = line.split('::')
            labels[line[0]] = int(line[1][1])
    return labels 

def docs(_dir=config.PATH_DATA_EN):
    """Processes and stores XML files to dictionary: name -> text"""
    X = {}
    list_files = readListDir(_dir)
    for file in list_files:
        if(file != 'truth.txt'):
            doc =ET.parse(_dir+"/"+file);
            fname = file.split(".")
            fname = fname[0]
            root = doc.getroot()
            X[fname] = []
            for child in root.iter('document'):
                X[fname].append(child.text)
            X[fname] = ' '.join(X[fname])
    return X


def exportTest(path):
    """ path: path to dataset folder 
        returns documents parsed raw in a list and their indexs """
    X = docs(path)
    final_texts=[]
    cnt = 0
    name_idx = list(X.keys())
    for k,v in X.items():
        text = X[k]
        final_texts.append(text)        
    return final_texts,name_idx

def export(path,path_truth):
    """Processes and returns dictionary: name -> text and labels """
    X = docs(path)
    Y = readLabels(path_truth)
    return X,Y

def exportMerged():
    """Processes and returns dictionary: name -> text and labels """
    X_en = docs(config.PATH_DATA_EN)
    Y_en = readLabels(config.PATH_DATA_EN_TRUTH)
    X_es = docs(config.PATH_DATA_ES)
    Y_es = readLabels(config.PATH_DATA_ES_TRUTH)    
    X = X_en.update(X_es)
    y = Y_en.update(Y_es)
    print(len(X_en.values()))
    return X_en,Y_en
    
def exportCSV():
    """Processes and exports CSV dictionary: name -> text and labels """
    X = docs()
    Y = readLabels()
    train,test = train_test_split(list(X.keys()), test_size = 0.3, shuffle=True)
    with open('train.csv',mode='w',newline='', encoding='utf8') as f:
        f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in train:
            f.writerow([X[row],Y[row]])
    with open('test.csv',mode='w',newline='', encoding='utf8') as f:
        f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in test:
            f.writerow([X[row],Y[row]])  

if __name__ == "__main__":
    #exportCSV()
    print(3)