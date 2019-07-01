import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
import clothing_df as df

clust = AgglomerativeClustering(n_clusters=3).fit(df.wear_pics)


a = df.wear_pics.index
b = df.wearable_df.loc[a,:]
c = clust.labels_

b['category_group'] = b['masterCategory'].apply(lambda x: 0 if x =='Apparel' else (1 if x=='Accessories' else 2))
b['cluster_group'] = c
d = b[['category_group', 'cluster_group']]

d[d['category_group']==0].groupby('cluster_group').count()
d[d['category_group']==1].groupby('cluster_group').count()
d[d['category_group']==2].groupby('cluster_group').count()

