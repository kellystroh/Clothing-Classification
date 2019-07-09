import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from full_df import df
all_img = np.load('data/image_array.npz')['a']
bw_img = np.load('data/image_array.npz')['b']
print('data loaded')

#from subset_df import apparel_subset, accessory_subset, shoe_subset, wearable_subset
from sklearn.manifold import TSNE
### PLOT TSNE ---- ALL
for perplex in [30, 50, 75, 100]
    all_tsne_bw = TSNE(n_components=2, n_iter=1000, perplexity=perplex).fit_transform(bw_img)
    print('model fitted, perplex =', perplex)
    np.save('tsne-bw-aws.npy', all_tsne_bw)
    print('np saved')

all_tsne_col = TSNE(n_components=2, n_iter=1000, perplexity=perplex).fit_transform(bw_img)
print('model fitted, perplex =', perplex)
np.save('tsne-bw-aws.npy', all_tsne_col)
print('np saved')


'''
#generate plot
colors = {'Apparel':'grey','Accessories':'teal','Footwear':'red', 'Personal Care':'orange', 'Free Items':'yellow', 'Sporting Goods':'black', 'Home':'magenta'}
df['color'] = df.loc[:,'masterCategory'].apply(lambda x: colors[x])
fig4, ax = plt.subplots(figsize=(6,6))
ax.scatter(XX[:,0],XX[:,1], s=.2, color= df['color'])
fig4.savefig('scatterplot-TSNA.png')
'''
