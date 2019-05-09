#%%
import glob
import re

# For PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# For HCA
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

# For Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.colors

#%%

use_idf = True
stop_words = 'english'
max_features = 1000
n_components = 10

#%%

countVectorizer = TfidfVectorizer(max_features=max_features, use_idf=use_idf, stop_words=stop_words)
countMatrix1 = countVectorizer.fit_transform(chaps.token_str)

#%%

countMatrix = normalize(countMatrix1)
countMatrix = countMatrix.toarray()

#%%

pca = PCA(n_components=n_components)
projected = pca.fit_transform(countMatrix)

#%%
vocab = pd.DataFrame([(v, countVectorizer.vocabulary_[v]) for v in countVectorizer.vocabulary_], 
                    columns=['term_str', 'term_id'])
vocab = vocab.set_index('term_id').sort_index()
vocab.head(40)

#%%
COMPS = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))
COMPS.columns = ["PC{}".format(i) for i in COMPS.columns]
COMPS.index = vocab.term_str

#%%
COMPS['PC9'].sort_values(ascending=False).head(20)

#%%
SIMS = pdist(countMatrix, metric='cosine')

#%%
SIMS = np.nan_to_num(SIMS, copy = True)

np.unique(SIMS, return_counts=True)

#%%
TREE = sch.linkage(SIMS, method='ward')

#%%
def plot_tree(tree, labels):
    plt.figure()
    fig, axes = plt.subplots(figsize=(30, 550),dpi = 110)
    dendrogram = sch.dendrogram(tree, labels=labels, orientation="left", distance_sort=True)
    plt.tick_params(axis='both', which='major', labelsize=5)
#%%
plot_tree(TREE,chaps.index)

#%%

clear()

#%%
# =============================================================================
# Image Too Large: Try a subset
# =============================================================================
chaps_sub = chaps.sample(2600)

#%%
countMatrix2 = countVectorizer.fit_transform(chaps_sub.token_str)

#%%

countMatrix3 = normalize(countMatrix2)
countMatrix3 = countMatrix3.toarray()

#%%

pca = PCA(n_components=n_components)
projected = pca.fit_transform(countMatrix3)

#%%
vocab = pd.DataFrame([(v, countVectorizer.vocabulary_[v]) for v in countVectorizer.vocabulary_], 
                    columns=['term_str', 'term_id'])
vocab = vocab.set_index('term_id').sort_index()
vocab.head(40)

#%%
COMPS = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))
COMPS.columns = ["PC{}".format(i) for i in COMPS.columns]
COMPS.index = vocab.term_str

#%%
COMPS['PC9'].sort_values(ascending=False).head(20)

#%%
SIMS = pdist(countMatrix3, metric='cosine')

#%%
SIMS = np.nan_to_num(SIMS, copy = True)

np.unique(SIMS, return_counts=True)

#%%
TREE = sch.linkage(SIMS, method='ward')

#%%
def plot_tree2(tree, labels):
    plt.figure()
    fig, axes = plt.subplots(figsize=(30, 350),dpi = 160)
    dendrogram = sch.dendrogram(tree, labels=labels, orientation="left", distance_sort=True)
    plt.tick_params(axis='both', which='major', labelsize=7)
#%%
plot_tree2(TREE,chaps_sub.index)

#%%
clear()