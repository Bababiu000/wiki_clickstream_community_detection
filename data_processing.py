import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_cls_gz(filepath, min_frequency=10):
    df = pd.read_csv(filepath, encoding='utf8', compression='gzip', header=None, sep='\t', quotechar='"',
                     skip_blank_lines=True)
    df.columns = ['from', 'to', 'type', 'frequency']
    df = df[~df['type'].isin(['external', 'other-external'])]
    df.drop(df[df.frequency < min_frequency].index, inplace=True)
    df = df.sort_values(by=['from'])

    return df


def cls_to_dict(df):
    unique_terms = np.union1d(df['from'].unique(), df['to'].unique())

    dict_df = pd.DataFrame(unique_terms, columns=['term'])
    dict_df['id'] = dict_df.index

    return dict_df


def cls_to_csr(dict_df, df):
    # 为cls的词加索引
    # 1. merge from, find the row id
    from_merged = pd.merge(dict_df, df, left_on=['term'], right_on=['from'])

    # 2. merge to, find the col id
    to_merged = pd.merge(from_merged, dict_df, left_on=[
        'to'], right_on=['term'])

    n_terms = dict_df.shape[0]
    affinity_graph = sp.csr_matrix((to_merged['frequency'], (to_merged['id_x'], to_merged['id_y'])),
                                   shape=(n_terms, n_terms))

    return affinity_graph


def direct_cls_to_symm(direct_g):
    return direct_g + direct_g.T
