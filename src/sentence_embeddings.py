### relation extractor

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

from itertools import combinations
import operator
import pandas as pd
import string
import numpy as np
import tqdm
import multiprocessing as mp
import scipy.sparse as sps

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from scipy import sparse
from gensim.models.doc2vec import Doc2Vec, TaggedDocument,TaggedLineDocument
from gensim.utils import simple_preprocess
class documentEmbedder:

    """
    Core class describing sentence embedding methodology employed here.
    """
    
    def __init__(self, max_features = 10000, num_cpu = 8, dm = 1, pretrained_path = "doc2vec.bin", ndim = 512):
        
        self.max_features = max_features
        self.dm = dm
        self.pretrained_path = pretrained_path
        self.vocabulary = {}
        self.ndim = ndim
        self.model = None
        if num_cpu == "all":
            self.num_cpu = mp.cpu_count()
            
        else:
            self.num_cpu = num_cpu
    
    def fit(self, text_vector, b = None, refit = False):

        """
        Fit the model to a text vector.
        """
        
        if self.model is None and not refit:

            documents = [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in enumerate(text_vector.values.tolist())]
            self.model = Doc2Vec(vector_size=self.ndim, window=3, min_count=1, workers=self.num_cpu, dm = self.dm)
            self.model.build_vocab(documents)
            self.model.train(documents,
                             total_examples=self.model.corpus_count,
                             epochs=32)
            self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        
    def transform(self, text_vector):

        """
        Transform the data into suitable form.
        """
        
        printable = set(string.printable)
        final_matrix = np.zeros((len(text_vector),self.ndim))
        for enx, doc in enumerate(tqdm.tqdm(text_vector)):
            if len(doc) > 1:
                try:
                    vector = self.model.infer_vector(simple_preprocess(doc))
                    final_matrix[enx] = vector
                except:
                    ## invalid inference.
                    pass
                

        logging.info("Generated embeddings ({}) of shape {}".format(self.dm, final_matrix.shape))
        
        return sparse.csr_matrix(final_matrix)

    def get_feature_names(self):

        return [str(x)+"_"+str(self.dm) for x in list(range(self.ndim))]
    
    def fit_transform(self, text_vector, a2 = None):

        """
        A classifc fit-transform method.
        """
        
        self.fit(text_vector)
        return self.transform(text_vector)
        
if __name__ == "__main__":

    example_text = pd.read_csv("../data/counterfactuals/train.tsv", sep="\t")['text_a']
    
    rex = documentEmbedder(dm = 1)
    rex.fit(example_text)
    
    m = rex.transform(example_text)

    print("+"*100)
    m = rex.fit_transform(example_text)
    print(m)
    
#    rex2 = entityDetector()
#    X = rex2.fit_transform(example_text[0:30])
#    print(X)
