import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from full_df import df, all_img, bw_img
from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset

from sklearn.decomposition import PCA

full_pca = pca.fit_transform(bw_img)
masterCat_colors = {'Apparel':'#008B8B','Accessories':'#808080','Footwear':'#8B0000', 'Personal Care':'magenta', 'Free Items':'magenta', 'Sporting Goods':'magenta', 'Home':'magenta'}
df['colors'] = df['masterCategory'].apply(lambda x: masterCat_colors[x])
fig0, ax = plt.subplots(figsize=(6,6))
ax.scatter( *full_pca.T, s=.25 , color=df['colors'])
fig0.savefig('fig0.png')

