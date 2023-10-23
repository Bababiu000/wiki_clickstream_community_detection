import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def load_cls_gz(filepath, min_frequency=10):
    df = pd.read_csv(filepath, encoding='utf8', compression='gzip', header=None, sep='\t',
                     quotechar='"',
                     skip_blank_lines=True)
    df.columns = ['from', 'to', 'type', 'frequency']
    df[['from', 'to']] = df[['from', 'to']].astype(str)
    df = df[~df['type'].isin(['external', 'other-external'])]
    df.drop(df[df.frequency < min_frequency].index, inplace=True)
    df = df.sort_values(by=['from'])

    return df


def sensitive_words_filter(df):
    pattern = re.compile(r'.*(近平|太子党|團派|政治家族列表|反對動態清零政策運動|文化大革命|文革|习明泽|江泽民|薄熙来|周永康|段伟红|習遠平|共产党|共產黨|胡锦涛|四通桥|秦刚|坦克人|赵紫阳|刘晓波|第十四世达赖喇嘛|杨家将事件|沈梦雨_(活动人士)|巴拿马文件|六四|柴玲|王丹|封从德|吾爾開希|黄雀行動|中华人民共和国被禁|影帝温家宝|温家宝家族财富|沈栋|康华|官倒|民主运动|中华人民共和国宪法修正案|辱包|膜蛤|抗议|佳士事件).*')

    filtered_condiction = df['from'].apply(lambda x: bool(pattern.search(x))) | df['to'].apply(lambda x: bool(pattern.search(x)))
    filter_rows = df[filtered_condiction]
    unique_terms = np.union1d(filter_rows['from'].unique(), filter_rows['to'].unique())
    sensitive_words_df = pd.DataFrame(unique_terms, columns=['term'])
    sensitive_words_df.to_csv('sensitive_words.csv', index=False, header=False)

    df = df[~filtered_condiction]

    df.reset_index(drop=True, inplace=True)

    print(f"filtered items: {len(filter_rows)}")

    return df


def cls_to_dict(df):
    unique_terms = np.union1d(df['from'].unique(), df['to'].unique())

    dict_df = pd.DataFrame(unique_terms, columns=['term'])
    dict_df['id'] = dict_df.index

    print(f"节点数量：{len(dict_df)}")

    return dict_df


def cls_to_csr(dict_df, df):
    # 为cls的词加索引
    # 1. merge from, find the row id
    from_merged = pd.merge(dict_df, df, left_on=['term'], right_on=['from'])

    # 2. merge to, find the col id
    to_merged = pd.merge(from_merged, dict_df, left_on=['to'], right_on=['term'])

    n_terms = dict_df.shape[0]
    affinity_graph = sp.csr_matrix((to_merged['frequency'], (to_merged['id_x'], to_merged['id_y'])),
                                   shape=(n_terms, n_terms))

    return affinity_graph


def direct_cls_to_symm(direct_g):
    return direct_g + direct_g.T


def remove_small_components(matrix, min_component_size=20):
    num_nodes = matrix.shape[0]

    # 找出连通分量
    num_components, labels = connected_components(matrix, directed=True, connection='weak')

    # 找出节点数量小于n的节点的索引
    nodes_to_remove = [node for node in range(num_nodes) if
                       np.sum(labels == labels[node]) <= min_component_size]

    # 创建新的 CSR 矩阵，仅包含保留的节点
    remaining_nodes = [node for node in range(num_nodes) if node not in nodes_to_remove]
    new_matrix = matrix[remaining_nodes][:, remaining_nodes]

    num_components_after = len(np.unique(labels[remaining_nodes]))
    print("处理前连通分量数量: ", num_components)
    print("处理后连通分量数量: ", num_components_after)
    print(f"节点数量：{len(remaining_nodes)}")

    return new_matrix, nodes_to_remove
