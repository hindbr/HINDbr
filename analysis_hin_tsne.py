from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json

# Project Setting
PROJECT = 'gnome'


# Data
DATA_CSV = 'data/model_training/' + PROJECT + '_all.csv'

# HIN embedding vector
HIN_EMBEDDING_DIM = '128'
HIN_EMBEDDING_FILE = 'data/pretrained_embeddings/hin2vec/' + PROJECT + '_node_' + HIN_EMBEDDING_DIM + 'd_5n_4w_1280l.vec'
HIN_NODE_DICT = 'data/hin_node_dict/' + PROJECT + '_node.dict'

data_df = pd.read_csv(DATA_CSV)
hin_cols1 = ["bid1","pro1","com1","ver1","sev1","pri1"]
hin_cols2 = ["bid2","pro2","com2","ver2","sev2","pri2"]

# Load hin node2vec
node2vec = {}
with open(HIN_EMBEDDING_FILE) as f:
    first = True
    for line in f:
        if first:
            first = False
            continue
        line = line.strip()
        tokens = line.split(' ')
        node2vec[tokens[0]] = np.array(tokens[1:],dtype=float)


with open(HIN_NODE_DICT, 'r') as f:
    hin_node_dict = json.load(f)


pair_vectors = 1 * np.random.randn(len(data_df), int(HIN_EMBEDDING_DIM)*6)
for index, row in data_df.iterrows():
    vectors1 = np.array([])
    for hin in hin_cols1:
        if str(row[hin]) != 'nan':
            hin_node_id = hin_node_dict[str(row[hin])][0]
            vectors1 = np.concatenate([vectors1, node2vec[str(hin_node_id)]])
        else:
            vectors1 = np.concatenate([vectors1, node2vec[str(hin_node_id)]])
    vectors2 = np.array([])
    for hin in hin_cols2:
        if str(row[hin]) != 'nan':
            hin_node_id = hin_node_dict[str(row[hin])][0]
            vectors2 = np.concatenate([vectors2, node2vec[str(hin_node_id)]])
        else:
            vectors2 = np.concatenate([vectors2, node2vec[str(hin_node_id)]])
    pair_vectors[index] = vectors1-vectors2

feat_cols = ['feature' + str(i) for i in range(int(HIN_EMBEDDING_DIM)*6)]
df = pd.DataFrame(pair_vectors,columns=feat_cols)
df['y'] = data_df['is_duplicate']


# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

print('Processing...')
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
df_subset[['tsne-pca50-one','tsne-pca50-two','y']].to_csv('output/tsne_result/' + PROJECT + '_tsne_results.csv', index=False, quoting=1)

