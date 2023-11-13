import numpy as np
from sentence_transformers import SentenceTransformer


print(" Sentence bert is used ")
pre_pro_data = np.load('pre_pro_data.npy', allow_pickle=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
vector_data = model.encode(pre_pro_data)
# np.save('vector_data.npy', vector_data)