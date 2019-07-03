import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
import clothing_df_short as df


def clusterize(df, pic_df, num):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering
    clust = AgglomerativeClustering(n_clusters=num).fit(df)

    a = pic_df.index
    b = df.loc[a,:]
    c = clust.labels_
    return a, b, c
'''
    b['category_group'] = b['masterCategory'].apply(lambda x: 0 if x =='Apparel' else (1 if x=='Accessories' else 2))
    b['cluster_group'] = c
    d = b[['category_group', 'cluster_group']]

    return 'Apparel: ', d[d['category_group']==0].groupby('cluster_group').count()
    return 'Accessories: ', d[d['category_group']==1].groupby('cluster_group').count()
    return 'Footwear: ', d[d['category_group']==2].groupby('cluster_group').count()'''