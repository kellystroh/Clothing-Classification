import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from full_df import df, all_img, bw_img
from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

### MAKE SUBSET FOR APPAREL
apparel_df, apparel_img = apparel_subset(df, all_img)
BW_apparel_df, BW_apparel_img = apparel_subset(df, bw_img)

### MAKE SUBSET FOR ACCESSORIES
accessory_df, accessory_img = accessory_subset(df, all_img)
BW_accessory_df, BW_accessory_img = accessory_subset(df, bw_img)

### MAKE SUBSET FOR SHOES
#footwear_df, footwear_img = shoe_subset(df, all_img)
BW_footwear_df, BW_footwear_img = shoe_subset(df, bw_img)

pca = PCA(n_components=2)

### PLOT PCA --- APPAREL
#apparel_pca = pca.fit_transform(apparel_img[0:10000])
apparel_pca = pca.fit_transform(BW_apparel_img[0:10000])

apparel_subset_colors = {'Bottomwear':'teal','Topwear':'orange','Dress':'magenta', 'Saree':'grey', 'Loungewear and Nightwear':'cyan', 'Innerwear':'red', 'Apparel Set':'cyan', 'Socks':'cyan'  }
BW_apparel_df['colors'] = BW_apparel_df.loc[:,'subCategory'].apply(lambda x: apparel_subset_colors[x])

fig1, ax = plt.subplots(figsize=(12,6))
#ax[0].scatter( *BW_apparel_pca.T, s=.35 , color=apparel_df['colors'][0:10000])
ax.scatter( *apparel_pca.T, s=.35 , color=BW_apparel_df['colors'][0:10000])
fig1.savefig('fig1.png')
### PLOT PCA --- SHOES
#shoe_pca = pca.fit_transform(footwear_img[0:10000])
shoe_pca = pca.fit_transform(BW_footwear_img[0:10000])

footwear_subset_colors = {'Shoes':'#F4A460','Sandal':'#008B8B','Flip Flops':'#8B0000'}
BW_footwear_df['colors'] = BW_footwear_df.loc[:,'subCategory'].apply(lambda x: footwear_subset_colors[x])

fig2, ax = plt.subplots(figsize=(12,6))
#ax[0].scatter( *shoe_pca.T, s=.5 , color=footwear_df['colors'][0:10000])
ax.scatter( *shoe_pca.T, s=.5, color=BW_footwear_df['colors']);
fig2.savefig('fig2.png')
### PLOT PCA --- ACCESSORIES

'''
DISPLAY=:0.0 ssh -Y <server ip>
'''


'''

## clust = AgglomerativeClustering(n_clusters=3).fit(df.wear_pics[0:100])
clust = AgglomerativeClustering(n_clusters=3).fit(df.wear_pics)

## a = df.wear_pics[0:100].index
a = df.wear_pics.index
b = df.wearable_df.loc[a,:]
c = clust.labels_

b['category_group'] = b['masterCategory'].apply(lambda x: 0 if x =='Apparel' else (1 if x=='Accessories' else 2))
b['cluster_group'] = c
d = b[['category_group', 'cluster_group']]

d[d['category_group']==0].groupby('cluster_group').count() 
d[d['category_group']==1].groupby('cluster_group').count() 
d[d['category_group']==2].groupby('cluster_group').count()


'''
