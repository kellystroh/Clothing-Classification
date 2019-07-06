import numpy as np
import pandas as pd

from full_df import df, bw_img, all_img

#df = pd.read_csv('data/styles.csv', error_bad_lines=False)
#bw_img = np.load('data/bw_array.npy')
#all_img = np.load('data/image_array.npy')
print('data loaded')

from subset_df import top2_subset

df_top, bw_top_img = top2_subset(df.iloc[0:500], bw_img[0:500])
df_top2, top2_img = top2_subset(df.iloc[0:500], all_img[0:500])
print('data subsetted & truncated')

from sklearn.cluster import AgglomerativeClustering

### BW TOP2
two_clust_bw = AgglomerativeClustering(n_clusters=2).fit(bw_top_img)
df_top.loc[:,'Cluster_Number'] = two_clust_bw.labels_
df_top.loc[:, 'Category_Number'] = df_top['masterCategory'].apply(lambda x: 0 if x =='Apparel' else 1)
print('agglo model fitted')

clust_by_cat1 = []
for x in range(2):
    w = df_top[df_top['Category_Number']==x].groupby('Cluster_Number').count()['masterCategory'].values
    clust_by_cat1.append(w)

clust2_df = pd.DataFrame(clust_by_cat1, index=['Apparel','Accessories'], columns=['Cluster 1', 'Cluster 2'])
clust2_df.to_pickle("./agglo2_bw.pkl")
print('saved the pickle')


two_clust_color = AgglomerativeClustering(n_clusters=2).fit(top2_img)
df_top2.loc[:,'Cluster_Number'] = two_clust_color.labels_
df_top2.loc[:, 'Category_Number'] = df_top2['masterCategory'].apply(lambda x: 0 if x =='Apparel' else (1 if x=='Accessories' else 2))

clust_by_cat2 = []
for x in range(2):
    w = df_top2[df_top2['Category_Number']==x].groupby('Cluster_Number').count()['masterCategory'].values
    clust_by_cat2.append(w)

clust2_df = pd.DataFrame(clust_by_cat2, index=['Apparel','Accessories'], columns=['Cluster 1', 'Cluster 2'])
clust2_df.to_pickle("./agglo2_color.pkl")