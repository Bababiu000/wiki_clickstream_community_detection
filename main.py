from datetime import datetime
import numpy as np
from scipy.sparse.csgraph import shortest_path
from data_processing import load_cls_gz, cls_to_csr, cls_to_dict, direct_cls_to_symm, remove_small_components
from utils import generate_date_strings, save_df_to_mysql, sensitive_words_filter


# 计算距离矩阵
def get_distance_matrix(cls_matrix):
    # 使用 Dijkstra 算法计算最短路径
    weighted_adjacency_matrix = cls_matrix.copy()
    weighted_adjacency_matrix.data = np.ones_like(weighted_adjacency_matrix.data)

    distance_matrix = shortest_path(weighted_adjacency_matrix, directed=False, method='D', return_predecessors=False)

    return distance_matrix


# 计算dc
def select_dc(distance_matrix):
    n = np.shape(distance_matrix)[0]
    distance_array = np.reshape(distance_matrix, n * n)
    percent = 2.0 / 100
    position = int(n * (n - 1) * percent)
    dc = np.sort(distance_array)[position + n]
    return dc


# 计算局部密度
def get_local_density(distance_matrix, dc, symm_cls):
    n = np.shape(distance_matrix)[0]
    rhos = np.zeros(n)
    for i in range(n):
        # rhos[i] = np.where(distance_matrix[i, :] < dc)[0].shape[0] - 1
        nearest_neighbor_indices = np.where(distance_matrix[i, :] < dc)[0]
        rhos[i] = np.sum(symm_cls[i, nearest_neighbor_indices])
        # rhos[i] = np.sum(symm_cls[nearest_neighbor_indices])
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
    centers = np.argsort(-rho_and_delta)

    # condition_indices = np.where(deltas >= 4)[0]
    # # 使用这些索引来排序 rhos 并得到排序后的索引
    # sorted_indices = np.argsort(-rhos[condition_indices])
    # # 使用排序后的索引获取原始索引
    # centers = condition_indices[sorted_indices]

    count_inf = np.sum(np.isinf(deltas))
    if count_inf > k:
        centers = centers[:count_inf]
    else:
        centers = centers[:k]
    print(f"centers: {len(centers)}")
    return centers


# 聚类
def density_peal_cluster(rhos, centers, nearest_neighbor):
    n = np.shape(rhos)[0]
    labels = -1 * np.ones(n).astype(int)
    dc_dict_idx = -1 * np.ones(n).astype(int)

    # 给刚刚找出来的簇心编号0， 1， 2， 3 ......
    for i, center in enumerate(centers):
        labels[center] = i
        dc_dict_idx[center] = center

    # 再将每个点编上与其最近的高密度点相同的编号
    rhos_index = np.argsort(-rhos)
    for i, index in enumerate(rhos_index):
        if labels[index] == -1:
            labels[index] = labels[int(nearest_neighbor[index])]
            dc_dict_idx[index] = dc_dict_idx[int(nearest_neighbor[index])]
    return labels, dc_dict_idx


def get_center_distance(distance_matrix, dc_dict_idx):
    n = distance_matrix.shape[0]
    center_distance = np.zeros(n)
    for i, center in enumerate(dc_dict_idx):
        center_distance[i] = distance_matrix[i][dc_dict_idx[i]]
    return center_distance


if __name__ == '__main__':
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "colt1911",
        "database": "wiki_clickstream",
    }
    table_name = 'clickstream'
    min_freq = 100
    dates = generate_date_strings('2023-08', '2023-08')

    for date in dates:

        cls_filepath = f"data\\clickstream-zhwiki-{date}.tsv.gz"

        # 数据预处理，建立无向图
        cls_df = load_cls_gz(cls_filepath, min_freq)
        cls_dict_df = cls_to_dict(cls_df)
        direct_cls = cls_to_csr(cls_dict_df, cls_df)
        symm_cls = direct_cls_to_symm(direct_cls)
        # 删除节点数量小于n的分部，重设索引
        symm_cls, removed_node_indices = remove_small_components(symm_cls)
        cls_dict_df = cls_dict_df[~cls_dict_df['id'].isin(removed_node_indices)]
        cls_dict_df['id'] = range(len(cls_dict_df))

        # DPC聚类
        distance_matrix = get_distance_matrix(symm_cls)  # 计算距离矩阵
        dc = select_dc(distance_matrix)  # 计算dc
        rhos = get_local_density(distance_matrix, dc, symm_cls)  # 计算局部密度
        deltas, nearest_neighbor = get_deltas(distance_matrix, rhos)  # 计算相对距离
        centers = find_cluster_centers(rhos, deltas, 200)  # 寻找簇心
        labels, dc_dict_idx = density_peal_cluster(rhos, centers, nearest_neighbor)
        center_dis = get_center_distance(distance_matrix, dc_dict_idx)

        # 处理和保存结果
        result_df = cls_dict_df.assign(density=rhos, dc_dict_idx=dc_dict_idx, label=labels, center_dis=center_dis,
                                       date=datetime.strptime(date, "%Y-%m").date())
        result_df = result_df.rename(columns={'term': 'name', 'id': 'dict_idx'})
        result_df = sensitive_words_filter(result_df)
        save_df_to_mysql(result_df, table_name, db_config)
