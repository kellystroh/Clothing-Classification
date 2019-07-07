import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from full_df import df, bw_img
from subset_df import wearable_subset

df, bw_img = wearable_subset(df, bw_img)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

full_pca = pca.fit_transform(bw_img)
masterCat_colors = {'Apparel':'#FF6347','Accessories':'#48D1CC','Footwear':'#FFA500'}
# {'Personal Care':'#808080', 'Free Items':'#808080', 'Sporting Goods':'#808080', 'Home':'#808080'}
df['colors'] = df['masterCategory'].apply(lambda x: masterCat_colors[x])
fig0, ax = plt.subplots(figsize=(6,6))
ax.scatter( *full_pca.T, s=.1 , color=df['colors'])
fig0.savefig('fig0.png')


