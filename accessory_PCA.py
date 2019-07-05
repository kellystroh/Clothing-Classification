import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from full_df import df, all_img, bw_img
from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset

from sklearn.decomposition import PCA


### MAKE SUBSET FOR ACCESSORIES
accessory_df, accessory_img = accessory_subset(df, all_img)
BW_accessory_df, BW_accessory_img = accessory_subset(df, bw_img)

pca = PCA(n_components=2)

### PLOT PCA --- ACCESSORIES
accessory_pca = pca.fit_transform(BW_accessory_img)

acc_sub_colors = {'Accessories':'orange', 'Bags':'#BA55D3', 'Belts':'purple', 'Cufflinks':'green', 'Eyewear':'#8B0000', 'Gloves':'orange',
       'Headwear':'cyan', 'Jewellery':'#008B8B', 'Mufflers':'orange', 'Perfumes':'orange', 'Scarves':'blue',
       'Shoe Accessories':'orange', 'Socks':'blue', 'Sports Accessories':'orange', 'Stoles':'orange', 'Ties':'blue',
       'Umbrellas':'orange', 'Wallets':'#F08080', 'Watches':'#808080', 'Water Bottle':'orange'}
BW_accessory_df['colors'] = BW_accessory_df.loc[:,'subCategory'].apply(lambda x: acc_sub_colors[x])

fig3, ax = plt.subplots(figsize=(6,6))
#ax[0].scatter( *shoe_pca.T, s=.5 , color=footwear_df['colors'][0:10000])
ax.scatter( *accessory_pca.T, s=.2, color=BW_accessory_df['colors']);
fig3.savefig('fig3.png')