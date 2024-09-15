from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

app = Flask(__name__)

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )
    return conn

@app.route('/add_product', methods=['POST'])
def add_product():
    try:
        product_data = request.get_json()
        product_id = product_data.get('product_id')
        product_name = product_data.get('product_name')

        if not isinstance(product_id, int) or not isinstance(product_name, int):
            return jsonify({'error': 'Product ID and Product Name must be integers'}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO Products (id, name) VALUES (%s, %s)',
            (product_id, product_name)
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Product added successfully!'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        user_data = request.get_json()
        user_id = user_data.get('user_id')
        user_email = user_data.get('user_email')

        if not isinstance(user_id, int) or not isinstance(user_email, str):
            return jsonify({'error': 'User ID must be an int and user_email must be a string'}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            'INSERT INTO Products (id, email) VALUES (%s, %s)',
            (user_id, user_email)
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'User added successfully!'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == "__main__":
    app.run()