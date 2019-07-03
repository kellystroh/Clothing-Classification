import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from full_df import df, all_img, bw_img
from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset

from sklearn.decomposition import PCA

### MAKE SUBSET FOR SHOES
#footwear_df, footwear_img = shoe_subset(df, all_img)
BW_footwear_df, BW_footwear_img = shoe_subset(df, bw_img)

pca = PCA(n_components=2)

### PLOT PCA --- SHOES
#shoe_pca = pca.fit_transform(footwear_img[0:10000])
shoe_pca = pca.fit_transform(BW_footwear_img)

footwear_subset_colors = {'Shoes':'blue','Sandal':'grey','Flip Flops':'red'}
BW_footwear_df['colors'] = BW_footwear_df.loc[:,'subCategory'].apply(lambda x: footwear_subset_colors[x])

fig2, ax = plt.subplots(figsize=(6,6))
#ax[0].scatter( *shoe_pca.T, s=.5 , color=footwear_df['colors'][0:10000])
ax.scatter( *shoe_pca.T, s=.2, color=BW_footwear_df['colors']);
fig2.savefig('fig2.png')