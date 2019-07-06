import numpy as np
import pandas as pd
df = pd.read_csv('data/styles.csv', error_bad_lines=False)
bw_img = np.load('data/bw_array.npy')
all_img = np.load('data/image_array.npy')
print('data loaded')
from subset_df import wearable_subset, top2_subset

df_wear, bw_wear_img = wearable_subset(df, bw_img)
df_wear2, wear_img = wearable_subset(df, all_img)
print('data subsetted')

df_wear = df_wear.iloc[0:1000, :]
bw_wear_img = bw_wear_img[0:1000, :]
df_wear2 = df_wear2.iloc[0:1000, :]
wear_img = wear_img[0:1000, :]
print('data truncated')
from sklearn.cluster import AgglomerativeClustering

### BW WEARABLE
three_clust_bw = AgglomerativeClustering(n_clusters=3).fit(bw_wear_img)
print('agglo model fitted')

df_wear.loc[:,'Cluster_Number'] = three_clust_bw.labels_
df_wear.loc[:, 'Category_Number'] = df_wear['masterCategory'].apply(lambda x: 0 if x =='Apparel' else (1 if x=='Accessories' else 2))
clust_by_cat1 = []
for x in range(3):
    w = df_wear[df_wear['Category_Number']==x].groupby('Cluster_Number').count()['masterCategory'].values
    clust_by_cat1.append(w)
print('yay')

clust3_bw_df = pd.DataFrame(clust_by_cat1, index=['Apparel','Accessories', 'Footwear'], columns=['Cluster 1', 'Cluster 2', 'Cluster 3'])
clust3_bw_df.to_pickle("agglo3_bw.pkl")
print('pickle saved')

### COLOR WEARABLE
three_clust_color = AgglomerativeClustering(n_clusters=3).fit(wear_img)
df_wear2.loc[:,'Cluster_Number'] = three_clust_color.labels_
df_wear2.loc[:, 'Category_Number'] = df_wear2['masterCategory'].apply(lambda x: 0 if x =='Apparel' else (1 if x=='Accessories' else 2))

clust_by_cat2 = []
for x in range(3):
    w = df_wear2[df_wear2['Category_Number']==x].groupby('Cluster_Number').count()['masterCategory'].values
    clust_by_cat2.append(w)

clust3_df = pd.DataFrame(clust_by_cat1, index=['Apparel','Accessories', 'Footwear'], columns=['Cluster 1', 'Cluster 2', 'Cluster 3'])
clust3_df.to_pickle("agglo3_color.pkl")