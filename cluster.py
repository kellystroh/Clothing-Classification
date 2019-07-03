import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from full_df import df, all_img, bw_img
from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

### ALL


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
apparel_pca = pca.fit_transform(BW_apparel_img)

apparel_subset_colors = {'Bottomwear':'#808080','Topwear':'#008B8B','Dress':'#8B0000', 'Saree':'orange', 'Loungewear and Nightwear':'#F08080', 'Innerwear':'#BA55D3', 'Apparel Set':'orange', 'Socks':'orange'  }
BW_apparel_df['colors'] = BW_apparel_df.loc[:,'subCategory'].apply(lambda x: apparel_subset_colors[x])

fig1, ax = plt.subplots(figsize=(12,6))
#ax[0].scatter( *BW_apparel_pca.T, s=.35 , color=apparel_df['colors'][0:10000])
ax.scatter( *apparel_pca.T, s=.35 , color=BW_apparel_df['colors'])
fig1.savefig('fig1.png')

### PLOT PCA --- SHOES
#shoe_pca = pca.fit_transform(footwear_img[0:10000])
shoe_pca = pca.fit_transform(BW_footwear_img)

footwear_subset_colors = {'Shoes':'#808080','Sandal':'#8B0000','Flip Flops':'#BA55D3'}
BW_footwear_df['colors'] = BW_footwear_df.loc[:,'subCategory'].apply(lambda x: footwear_subset_colors[x])

fig2, ax = plt.subplots(figsize=(12,6))
#ax[0].scatter( *shoe_pca.T, s=.5 , color=footwear_df['colors'][0:10000])
ax.scatter( *shoe_pca.T, s=.5, color=BW_footwear_df['colors']);
fig2.savefig('fig2.png')


### PLOT PCA --- ACCESSORIES
accessory_pca = pca.fit_transform(BW_accessory_img)

acc_sub_colors = {'Accessories':'orange', 'Bags':'#BA55D3', 'Belts':'purple', 'Cufflinks':'green', 'Eyewear':'#8B0000', 'Gloves':'orange',
       'Headwear':'cyan', 'Jewellery':'#008B8B', 'Mufflers':'orange', 'Perfumes':'orange', 'Scarves':'blue',
       'Shoe Accessories':'orange', 'Socks':'blue', 'Sports Accessories':'orange', 'Stoles':'orange', 'Ties':'blue',
       'Umbrellas':'orange', 'Wallets':'#F08080', 'Watches':'#808080', 'Water Bottle':'orange'}
BW_accessory_df['colors'] = BW_accessory_df.loc[:,'subCategory'].apply(lambda x: acc_sub_colors[x])

fig3, ax = plt.subplots(figsize=(12,6))
#ax[0].scatter( *shoe_pca.T, s=.5 , color=footwear_df['colors'][0:10000])
ax.scatter( *shoe_pca.T, s=.5, color=BW_footwear_df['colors']);
fig3.savefig('fig3.png')


### PLOT TSNE ---- ALL
all_tsne = TSNE(n_components=2, n_iter=1000, perplexity=100).fit_transform(bw_img)
colors = {'Apparel':'orange','Accessories':'teal','Footwear':'red', 'Personal Care':'grey', 'Free Items':'yellow', 'Sporting Goods':'black', 'Home':'magenta'}
df['color'] = df.loc[:,'masterCategory'].apply(lambda x: colors[x])
fig4, ax = plt.subplots(figsize=(12,6))
ax.scatter(XX[:,0],XX[:,1], s=.2, color= df['color'])
fig4.savefig('fig4.png')
