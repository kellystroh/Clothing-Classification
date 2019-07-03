import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from full_df import df, all_img, bw_img
from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset

from sklearn.decomposition import PCA

### MAKE SUBSET FOR APPAREL
apparel_df, apparel_img = apparel_subset(df, all_img)
BW_apparel_df, BW_apparel_img = apparel_subset(df, bw_img)

pca = PCA(n_components=2)

### PLOT PCA --- APPAREL
#apparel_pca = pca.fit_transform(apparel_img[0:10000])
apparel_pca = pca.fit_transform(BW_apparel_img)

apparel_subset_colors = {'Bottomwear':'#808080','Topwear':'#008B8B','Dress':'#8B0000', 'Saree':'orange', 'Loungewear and Nightwear':'#F08080', 'Innerwear':'#BA55D3', 'Apparel Set':'orange', 'Socks':'orange'  }
BW_apparel_df['colors'] = BW_apparel_df.loc[:,'subCategory'].apply(lambda x: apparel_subset_colors[x])

fig1, ax = plt.subplots(figsize=(6,6))
#ax[0].scatter( *BW_apparel_pca.T, s=.35 , color=apparel_df['colors'][0:10000])
ax.scatter( *apparel_pca.T, s=.35 , color=BW_apparel_df['colors'])
fig1.savefig('fig1.png')