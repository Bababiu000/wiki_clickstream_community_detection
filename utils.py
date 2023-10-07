import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine


def generate_date_strings(start_date_str, end_date_str):
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m")
        end_date = datetime.strptime(end_date_str, "%Y-%m")
    except ValueError:
        return ["Invalid date format"]

    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_string = current_date.strftime("%Y-%m")
        date_list.append(date_string)
        # 使用 relativedelta 来增加一个月
        current_date += relativedelta(months=1)

    return date_list


def save_df_to_mysql(df, table_name, db_config):
    try:
        # 创建数据库连接
        engine = create_engine(
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
        connection = engine.connect()
        delete_query = f"DELETE FROM {table_name} WHERE date = DATE('{df['date'][0]}')\n"
        connection.execute(delete_query)
        connection.close()
        # 将 DataFrame 存储到 MySQL 数据库表
        df.to_sql(table_name, con=engine, if_exists='append', index=False)

        print(f"DataFrame 存储到 {table_name} 表成功")
    except Exception as e:
        traceback.print_exc()


def sensitive_words_filter(df):
    filtered_rows = df[df['name'].isin(['习近平', '六四事件', '文化大革命'])]
    df = df[~df['dc_dict_idx'].isin(filtered_rows['dc_dict_idx'])]

    df.reset_index(drop=True, inplace=True)

    return df


def remove_small_components(distance_matrix, min_component_size=2):
    # 使用距离矩阵创建图
    G = nx.Graph()
    num_nodes = distance_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distance_matrix[i, j] != 0:
                G.add_edge(i, j, weight=distance_matrix[i, j])

    # 找出连通分量
    _, components = connected_components(csr_matrix(distance_matrix), directed=False)

    # 清除节点数量小于 min_component_size 的连通分量
    components_to_remove = set()
    for component_id in range(max(components) + 1):
        component_size = np.sum(components == component_id)
        if component_size < min_component_size:
            components_to_remove.add(component_id)

    # 根据需要清除节点
    nodes_to_remove = [node for node, component_id in enumerate(components) if component_id in components_to_remove]
    G.remove_nodes_from(nodes_to_remove)

    # 重新计算距离矩阵
    updated_distance_matrix = np.zeros_like(distance_matrix)
    for edge in G.edges(data=True):
        i, j, weight = edge
        updated_distance_matrix[i, j] = weight
        updated_distance_matrix[j, i] = weight

    return updated_distance_matrix


# 示例距离矩阵
distance_matrix = np.array([[0, 1, 2, 0],
                            [1, 0, 0, 0],
                            [2, 0, 0, 1],
                            [0, 0, 1, 0]])

# 清除节点数量小于2的分部
new_distance_matrix = remove_small_components(distance_matrix, min_component_size=2)

print("原始距离矩阵:")
print(distance_matrix)

print("清除后的距离矩阵:")
print(new_distance_matrix)
