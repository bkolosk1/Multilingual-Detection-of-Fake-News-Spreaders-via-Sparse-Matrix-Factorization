# Multilingual-Detection-of-Fake-News-Spreaders-via-Sparse-Matrix-Factorization

# Abstract
Fake news is an emerging problem in online news and social media. Efficient detection of fake news spreaders and spurious accounts across multiple languages is becoming an interesting research problem, and is the key focus of this paper. Our proposed solution to PAN 2020 fake news spreaders challenge models the accounts responsible for spreading the fake news by accounting for different types of textual features, decomposed via sparse matrix factorization, to obtain easy-to-learn-from, compact representations, including the information from multiple languages. The key contribution of this work is the exploration of how powerful and scalable matrix factorization-based classification can be in a multilingual setting, where the learner is presented with the data from multiple languages simultaneously. Finally, we explore the joint latent space, where patterns from individual languages are maintained. The proposed approach scored second on the 2020 PAN shared task for identification of fake news spreaders.

# Prerequired dependencies

``` 
nltk==3.5
pandas==1.0.5
matplotlib==3.2.2
seaborn==0.10.1
scipy==1.5.0
gensim==3.8.3
numpy==1.18.5
tqdm==4.47.0
scikit_learn==0.23.2
simpletransformers==0.48.14
umap==0.1.1 
```

# Data

The dataset for this task is provided by the PAN workshop organizators of the CLEF'20 conference. The link to the dataset can be found on the following link:

`` https://zenodo.org/record/4039435#.X4lVO08zaEI ``

# Model training

### To reproduce our models use the following python script

`` python src/run.py <OPTION> ``

Following models are provided

```
  --mono       TIRA monolingual model
  --multi      TIRA multilingual model
  --fullbatch  Monolingual Fullbatch model
  --extended   Monolingual extended grid model
```

# Evaluation

### To evaluate on a dataset use the following scripts

``cd src`` \
``python evaluate.py -i "IN" -o "OUT" `` \
### Default line to execute the evaluation is:
``python evaluate.py -i ../train_data -o ../out``

Arguments of the script:

```
  -i  IN       Directory where the input data is to be found 
  -o OUT       Directory where the outputs will be stored
```

# Contribution

### This code was develobed by Boshko Koloski & Blaž Škrlj


# Citation

If you use our code please cite our work. 

```
@InProceedings{koloski:2020a,
  author =              {Bo{\v s}ko Koloski and Senja Pollak and Bla{\v z} {\v S}krlj},
  booktitle =           {{CLEF 2020 Labs and Workshops, Notebook Papers}},
  crossref =            {pan:2020},
  editor =              {Linda Cappellato and Carsten Eickhoff and Nicola Ferro and Aur{\'e}lie N{\'e}v{\'e}ol},
  month =               sep,
  publisher =           {CEUR-WS.org},
  title =               {{Multilingual Detection of Fake News Spreaders via Sparse Matrix Factorization---Notebook for PAN at CLEF 2020}},
  url =                 {},
  year =                2020
}
```


# Aknowledgements

The work of the last author was funded by the Slovenian Research Agency through a
young researcher grant. The work of other authors was supported by the Slovenian
Research Agency (ARRS) core research programme Knowledge Technologies
(P2-0103), an ARRS funded research project Semantic Data Mining for Linked Open
Data (financed under the ERC Complementary Scheme, N2-0078) and European
Unions Horizon 2020 research and innovation programme under grant agreement No ´
825153, project EMBEDDIA (Cross-Lingual Embeddings for Less-Represented
Languages in European News Media).