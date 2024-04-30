import pandas as pd
import mysql.connector
import os
from utils.utils import load_credentials


def fetch_target():
    # Connect to the database
    load_credentials()
    connection = mysql.connector.connect(
        host=os.environ["host"],
        port=os.environ["port"],
        user=os.environ["username"],
        passwd=os.environ["password"],
        database=os.environ["database"]
    )

    if connection.is_connected():
        print('Connected to the MySQL database')
    else:
        print('Failed to connect to the MySQL database')

    sql_query = """
                WITH subquery AS (SELECT order_lines.*, orders.dataobra, orders.no
                    FROM order_lines
                    LEFT JOIN orders 
                        ON orders.id = order_lines.order_id
                    )
                SELECT
                    -- SUM(edebito) as edebito, 
                    SUM(ettdeb) as ettdeb,
                    subquery.product_id, 
                    family_id,
                    clients.tipo as tipo,
                    DATE_FORMAT(dataobra, '%Y-%m-%d') AS date
                FROM subquery
                LEFT JOIN products
                    ON products.id = subquery.product_id
                LEFT JOIN clients
                    ON clients.id = subquery.no
                GROUP BY subquery.product_id
                ORDER BY dataobra DESC
                """   
    daily_data = pd.read_sql(sql_query, connection)

    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print('Connection closed')

    return (daily_data)

def fetch_discount():
    # Connect to the database
    load_credentials()
    connection = mysql.connector.connect(
        host=os.environ["host"],
        port=os.environ["port"],
        user=os.environ["username"],
        passwd=os.environ["password"],
        database=os.environ["database"]
    )

    if connection.is_connected():
        print('Connected to the MySQL database')
    else:
        print('Failed to connect to the MySQL database')

    sql_query = """
                WITH subquery AS (SELECT order_lines.*, orders.dataobra, orders.no
                    FROM order_lines
                    LEFT JOIN orders 
                        ON orders.id = order_lines.order_id
                    )
                SELECT
                    -- SUM(edebito) as edebito, 
                    SUM(desconto*ettdeb) as desconto,
                    subquery.product_id, 
                    family_id,
                    clients.tipo as tipo,
                    DATE_FORMAT(dataobra, '%Y-%m-%d') AS date
                FROM subquery
                LEFT JOIN products
                    ON products.id = subquery.product_id
                LEFT JOIN clients
                    ON clients.id = subquery.no
                WHERE desconto <> 0
                GROUP BY subquery.product_id
                ORDER BY dataobra DESC
                """   
    # count discount <> zero
    discount_data = pd.read_sql(sql_query, connection)

    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print('Connection closed')

    return (discount_data)

def fetch_orders():
    load_credentials()
    # Connect to the database
    connection = mysql.connector.connect(
        host=os.environ["host"],
        port=os.environ["port"],
        user=os.environ["username"],
        passwd=os.environ["password"],
        database=os.environ["database"]
    )

    if connection.is_connected():
        print('Connected to the MySQL database')
    else:
        print('Failed to connect to the MySQL database')

    sql_query = """
                WITH subquery AS (SELECT order_lines.*, orders.dataobra, orders.no
                    FROM order_lines
                    LEFT JOIN orders 
                        ON orders.id = order_lines.order_id
                    )
                SELECT
                    -- SUM(edebito) as edebito, 
                    COUNT(order_id) as orders,
                    subquery.product_id, 
                    family_id,
                    clients.tipo as tipo,
                    DATE_FORMAT(dataobra, '%Y-%m-%d') AS date
                FROM subquery
                LEFT JOIN products
                    ON products.id = subquery.product_id
                LEFT JOIN clients
                    ON clients.id = subquery.no
                GROUP BY subquery.product_id
                ORDER BY dataobra DESC
                """   
    orders_data = pd.read_sql(sql_query, connection)

    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print('Connection closed')

    return (orders_data)

def fetch_quantities():
    load_credentials()
    # Connect to the database
    connection = mysql.connector.connect(
        host=os.environ["host"],
        port=os.environ["port"],
        user=os.environ["username"],
        passwd=os.environ["password"],
        database=os.environ["database"]
    )

    if connection.is_connected():
        print('Connected to the MySQL database')
    else:
        print('Failed to connect to the MySQL database')

    sql_query = """
                WITH subquery AS (SELECT order_lines.*, orders.dataobra, orders.no
                    FROM order_lines
                    LEFT JOIN orders 
                        ON orders.id = order_lines.order_id
                    )
                SELECT
                    -- SUM(edebito) as edebito, 
                    SUM(qtt) as quantity,
                    subquery.product_id, 
                    family_id,
                    clients.tipo as tipo,
                    DATE_FORMAT(dataobra, '%Y-%m-%d') AS date
                FROM subquery
                LEFT JOIN products
                    ON products.id = subquery.product_id
                LEFT JOIN clients
                    ON clients.id = subquery.no
                WHERE desconto <> 0
                GROUP BY subquery.product_id
                ORDER BY dataobra DESC
                """   
    # count discount <> zero
    qtt_data = pd.read_sql(sql_query, connection)

    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print('Connection closed')

    return (qtt_data)