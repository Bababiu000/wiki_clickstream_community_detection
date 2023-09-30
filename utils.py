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
        print(f"{str(e)}")