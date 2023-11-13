import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

given_data = pd.read_csv('data/articles.csv', encoding='latin-1')
## To get complete data
data_1 = given_data.Full_Article
## to display given data
print(data_1)
lab = given_data.Article_Type.factorize()
## to display label
lab_fin = lab[0]
# np.save('lab_fin.npy',lab_fin)
art_type = lab[1]
print(lab)
## data preprocessing
pre_pro_data = []
for ind,words in enumerate(data_1):
    print(ind)
    ## to find stop words and remove stop words
    stop_words = set(stopwords.words('english'))
    rem_stop = [w for w in words.split() if not w in stop_words]
    ## for stemming process
    stemmer = PorterStemmer()
    Stem_rem = [stemmer.stem(word) for word in rem_stop]
    rem_short = []
    for i in Stem_rem:
        if len(i) >= 3:  # removing short word
            rem_short.append(i)
    pre_pro_data.append((" ".join(rem_short)).strip())
print(pre_pro_data)
# np.save('pre_pro_data.npy', pre_pro_data)