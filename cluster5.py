import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from full_df import df, all_img, bw_img
from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

wear_df, wear_img = wearable_subset(df, bw_img)
wear_pca = pca.fit_transform(wear_img)
wearable_colors = {'Apparel':'#008B8B','Accessories':'orange','Footwear':'#8B0000'}
wear_df['colors'] = wear_df['masterCategory'].apply(lambda x: wearable_colors[x])
fig5, ax = plt.subplots(figsize=(6,6))
ax.scatter( *wear_pca.T, s=.1 , color=wear_df['colors'])
fig5.savefig('fig5b.png')
