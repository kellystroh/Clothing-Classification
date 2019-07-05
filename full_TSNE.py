import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from full_df import df, all_img, bw_img
from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset


from sklearn.manifold import TSNE

### PLOT TSNE ---- ALL
all_tsne = TSNE(n_components=2, n_iter=1000, perplexity=100).fit_transform(bw_img[0:100])
colors = {'Apparel':'grey','Accessories':'teal','Footwear':'red', 'Personal Care':'orange', 'Free Items':'yellow', 'Sporting Goods':'black', 'Home':'magenta'}
df['color'] = df.loc[:,'masterCategory'].apply(lambda x: colors[x])
fig4, ax = plt.subplots(figsize=(6,6))
ax.scatter(XX[:,0],XX[:,1], s=.2, color= df['color'])
fig4.savefig('scatterplot-TSNA.png')
