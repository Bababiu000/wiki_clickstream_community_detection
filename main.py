import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import umap.umap_ as umap
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

from config import DEV_DB_CONFIG
from data_processing import load_cls_gz, sensitive_words_filter, cls_to_csr, cls_to_dict, direct_cls_to_symm, \
    remove_small_components
from utils import generate_date_strings, save_df_to_mysql


# 计算距离矩阵
def get_distance_matrix(cls_matrix):
    # 使用 Dijkstra 算法计算最短路径
    weighted_adjacency_matrix = cls_matrix.copy()
    weighted_adjacency_matrix.data = np.ones_like(weighted_adjacency_matrix.data)

    distance_matrix = shortest_path(weighted_adjacency_matrix, directed=False, method='D',
                                    return_predecessors=False)

    return distance_matrix


# 计算局部密度
def get_local_density(distance_matrix, symm_cls):
    n = np.shape(distance_matrix)[0]
    rhos = np.zeros(n)
    for i in range(n):
        rhos[i] = np.sum(symm_cls[i, :])

    return rhos


# 计算相对距离
def get_deltas(distance_matrix, rhos):
    n = np.shape(distance_matrix)[0]
    deltas = np.zeros(n)  # 对于i，i与其密度更大的点的中最近点的距离
    nearest_neighbor = np.zeros(n)  # 对于i，比这个点密度更大的点的中最近点的索引
    rhos_index = np.argsort(-rhos)  # 得到密度ρ从大到小的排序的索引

    for i, index in enumerate(rhos_index):
        # i是序号，index是rhos_index[i]，是第i大的ρ的索引，这里第0大是最大的。
        # index是点的索引，rhos[index]和deltas[index]是第index个点的ρ和δ
        if i == 0:
            continue

        # 对于密度更大的点，计算最小可达距离
        higher_rho_indices = rhos_index[:i]
        min_distance = np.min(distance_matrix[index, higher_rho_indices])

        # 找到最近邻居的索引
        nearest_neighbor_index = np.argmin(distance_matrix[index, higher_rho_indices])
        nearest_neighbor_index = higher_rho_indices[nearest_neighbor_index]

        # 将结果存储在相应的数组中
        deltas[index] = min_distance
        nearest_neighbor[index] = nearest_neighbor_index.astype(int)

    deltas[rhos_index[0]] = np.max(deltas)

    return deltas, nearest_neighbor


# 寻找聚类中心
def find_cluster_centers(rhos, deltas, k):
    rho_and_delta = rhos * deltas
    centers = np.argsort(-rho_and_delta)  # 按 rhos * deltas 的值进行降序排序

    centers = centers[:k]

    centers = sorted(centers, key=lambda index: -rhos[index])   # 将centers 按 rhos 的值进行降序排序

    print(f"centers: {len(centers)}")

    return centers


def plot_decision_graph(rhos, deltas, centers, dict_df):
    # 将delta中无穷大值替换为最大值加1
    deltas_copy = np.copy(deltas)
    max_non_inf = np.max(deltas_copy[np.isfinite(deltas_copy)])  # 找到除了无穷大值之外的最大值
    deltas_copy[np.isinf(deltas_copy)] = max_non_inf + 1  # 将无穷大值替换为最大值加1

    plt.rcParams['font.family'] = ['SimHei']
    plt.figure(figsize=(8, 6))
    plt.scatter(rhos, deltas_copy, c='blue', s=20, alpha=0.5)

    # 标记所有聚类中心为红色点
    for index, center in enumerate(centers):
        if index < 300:
            plt.scatter(rhos[center], deltas_copy[center], c='red', marker='o', s=20)

        # if dict_df.loc[center, 'term'] == '奧本海默_(電影)':
        # if dict_df.loc[center, 'term'] == 'Oppenheimer_(film)':
        # if dict_df.loc[center, 'term'] == 'Taylor_Swift':
        if dict_df.loc[center, 'term'] == '長相思_(電視劇)':
            # 计算重要性排名与节点总数的比值
            ratio = "{:.6f}".format((index + 1) / len(rhos))
            plt.text(rhos[center], deltas_copy[center], f"{dict_df.loc[center, 'term']} {ratio}", fontsize=10,
                     ha='center', va='bottom')

    plt.title(date)
    plt.xlabel('局部密度')
    plt.ylabel('相对距离')
    plt.show()


# 对非聚类中心数据点进行归类
def density_peal_cluster(rhos, centers, nearest_neighbor):
    n = np.shape(rhos)[0]
    labels = -1 * np.ones(n).astype(int)
    dc_dict_idx = -1 * np.ones(n).astype(int)

    # 给刚刚找出来的簇心编号 1， 2， 3 ......
    for i, center in enumerate(centers):
        labels[center] = i + 1
        dc_dict_idx[center] = center

    # 再将每个点编上与其最近的高密度点相同的编号
    rhos_index = np.argsort(-rhos)
    for i, index in enumerate(rhos_index):
        if labels[index] == -1:
            labels[index] = labels[int(nearest_neighbor[index])]
            dc_dict_idx[index] = dc_dict_idx[int(nearest_neighbor[index])]

    return labels, dc_dict_idx


def get_directed_edges(adjacency_matrix):
    rows, cols = adjacency_matrix.nonzero()
    edges = []
    for i in range(len(rows)):
        source_node = rows[i]
        target_node = cols[i]
        weight = adjacency_matrix[source_node, target_node]  # 获取相应边的权重
        distance = 1
        edges.append((source_node, target_node, weight, distance))

    return edges


def get_aggregate_edges(distance_matrix, centers, edges):
    sub_matrix = distance_matrix[np.ix_(centers, centers)]
    rows, cols = sub_matrix.nonzero()
    # 遍历子矩阵，找出距离大于1的边
    for i in range(len(rows)):
        distance = sub_matrix[rows[i], cols[i]]
        if 1 < distance < np.inf:
            source_node = centers[rows[i]]  # 映射回原始索引
            target_node = centers[cols[i]]  # 映射回原始索引
            edges.append((source_node, target_node, None, distance))

    return edges


if __name__ == '__main__':
    db_config = DEV_DB_CONFIG
    min_freq = 100
    center_num = 300
    lang = 'zh'
    dates = generate_date_strings('2023-07', '2023-07')

    for date in dates:
        try:

            print(date)

            start_time = time.time()

            cls_filepath = f"data\\clickstream-{lang}wiki-{date}.tsv.gz"

            # 数据预处理，建立有向图
            cls_df = load_cls_gz(cls_filepath, min_freq)
            cls_df = sensitive_words_filter(cls_df, lang)
            cls_dict_df = cls_to_dict(cls_df)
            direct_cls = cls_to_csr(cls_dict_df, cls_df)
            # 删除有向图中节点数量小于n的弱连通分量，重设索引
            direct_cls, removed_node_indices = remove_small_components(direct_cls)
            cls_dict_df = cls_dict_df[~cls_dict_df['id'].isin(removed_node_indices)]
            cls_dict_df['id'] = range(len(cls_dict_df))
            cls_dict_df.reset_index(drop=True, inplace=True)
            symm_cls = direct_cls_to_symm(direct_cls)  # 有向图转无向图
            # 使用umap对距离矩阵降维，得到二维坐标
            umap_emb = umap.UMAP(n_components=2).fit_transform(symm_cls)

            # DPC聚类
            distance_matrix = get_distance_matrix(symm_cls)  # 计算距离矩阵
            rhos = get_local_density(distance_matrix, symm_cls)  # 计算局部密度
            deltas, nearest_neighbor = get_deltas(distance_matrix, rhos)  # 计算相对距离
            centers = find_cluster_centers(rhos, deltas, center_num)  # 寻找簇心
            # plot_decision_graph(rhos, deltas, centers, cls_dict_df)  # 绘制决策图
            labels, dc_dict_idx = density_peal_cluster(rhos, centers, nearest_neighbor)

            # 处理和保存结果
            # 节点数据
            cls_node_df = cls_dict_df.assign(density=rhos, dc_dict_idx=dc_dict_idx, label=labels, x=umap_emb[:, 0],
                                             y=umap_emb[:, 1], date=datetime.strptime(date, "%Y-%m").date())
            cls_node_df = cls_node_df.rename(columns={'term': 'name', 'id': 'dict_idx'})
            cls_node_df['lang'] = lang
            save_df_to_mysql(cls_node_df, 'clickstream_node', db_config)

            # 直连边数据
            edge_list = get_directed_edges(direct_cls)
            # 聚合边数据
            # edge_list = get_aggregate_edges(distance_matrix, centers, edge_list)
            cls_edge_df = pd.DataFrame(edge_list, columns=['source_dict_idx', 'target_dict_idx', 'weight', 'distance'])
            cls_edge_df['date'] = datetime.strptime(date, "%Y-%m").date()
            cls_edge_df['lang'] = lang
            save_df_to_mysql(cls_edge_df, 'clickstream_edge', db_config)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"代码运行时间: {elapsed_time:.2f} 秒")

        except Exception as e:
            print(f"Error : {str(e)}")
            traceback.print_exc()
