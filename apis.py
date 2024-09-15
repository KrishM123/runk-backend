import logging
from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv
import psycopg2
import json
from urllib.parse import urlparse
import requests
from search import *

load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

def get_db_connection():
    database_url = os.getenv("POSTGRES_URL")
    
    if database_url:
        result = urlparse(database_url)
        conn = psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
        )
    else:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            database=os.getenv("POSTGRES_DATABASE"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            port=os.getenv("POSTGRES_PORT")
        )
    
    return conn

@app.route('/', methods=['GET'])
def home():
    app.logger.info("Home route accessed")
    return "Hello, World!"

@app.route('/add_product', methods=['POST'])
def add_product():
    app.logger.info("add_product route accessed")
    try:
        product_data = request.get_json()
        if not product_data:
            return jsonify({'error': 'No JSON data provided'}), 400

        product_name = product_data.get('product_name')

        if product_name is None:
            return jsonify({'error': 'Missing product_name'}), 400

        if not isinstance(product_name, str):
            return jsonify({'error': 'Product Name must be a string'}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO products (name) VALUES (%s) RETURNING id",
            (product_name,)
        )
        new_product_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'Product added successfully!', 'product_id': new_product_id}), 201

    except Exception as e:
        app.logger.error(f"Error in add_product: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_user', methods=['POST'])
def add_user():
    app.logger.info("add_user route accessed")
    try:
        user_data = request.get_json()
        user_id = user_data.get('user_id')
        user_email = user_data.get('user_email')
        profile = '50, 50, 50, 3, 4, 4, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 25, 50, 2, 3, 0, 3, 50, 50, 50, 50, 50, 50, 1, 0, 8, 2, 50, 50, 4, 1, 50, 50, 1, 6, 4'

        if not isinstance(user_id, int) or not isinstance(user_email, str):
            return jsonify({'error': 'User ID must be an int and user_email must be a string'}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (id, email, profile) VALUES (%s, %s, %s)",
            (user_id, user_email, profile)
        )
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'message': 'User added successfully!'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ranked_review', methods=['GET'])
def ranked_review():
    try:
        review_data = request.get_json()
        user_email = review_data.get('user_email')
        product_name = review_data.get('product_name')

        if not isinstance(user_email, str) or not isinstance(product_name, int):
            return jsonify({'error': 'User Email and Product Name must be strings'}), 400
        
        product_id = get_product_id_by_name(product_name)
        if not product_id:
            return jsonify({'message': None}), 201
        
        conn = get_db_connection()
        cur = conn.cursor()

        #TODO: Aly plz finish give me SORTED LIST OF (USER ID, CORRELATION)
        sorted_list_of_user_correlation = []
        
        user_ids = [user_id for user_id, _ in sorted_list_of_user_correlation]

        format_strings = ','.join(['%s'] * len(user_ids))

        cur.execute(f"""
            SELECT user-review.user_id, reviews.text
            FROM user-review
            JOIN reviews ON user-review.review_id = reviews.id
            WHERE user-review.user_id IN ({format_strings});
        """, tuple(user_ids))

        review_data = cur.fetchall()

        cur.close()
        conn.close()

        user_to_text = {user_id: text for user_id, text in review_data}

        result = [(user_to_text.get(user_id, None), int_val) for user_id, int_val in sorted_list_of_user_correlation]

        return jsonify({'message': result}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user_review', methods=['POST'])
def user_review():
    try:
        review_data = request.get_json()
        user_email = review_data.get('user_email')
        review_text = review_data.get('review_text')
        product_name = review_data.get('product_name')
        product_id = get_product_id_by_name(product_name)
        user_id = get_user_id_by_email(user_email)
        user_json_profile = convert_textvec_to_profilejson(get_embedding_by_id(user_id))
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        if not product_id:
            cur.execute(
                "INSERT INTO products (name) VALUES (%s) RETURNING id",
                (product_name,)
            )
            conn.commit()
        
        #TODO: Aly take store_pinecone(review_text, user_id, product_id). Return newly created profile
        
        updated_user_json_profile = user_json_profile
        cur.execute(
            "UPDATE users SET profile = %s WHERE id = %s",
            (updated_user_json_profile, user_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'message': 'Complete'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rank_products', methods=['GET'])
def rank_products():
    try:
        product_data = request.get_json()
        user_email = product_data.get('user_email')
        product_id_list = product_data.get('product_id_list')
        user_id = get_user_id_by_email(user_email)
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Fetch product names for the given product_id_list
        sql_query = """
        SELECT
            product_id,
            product_name
        FROM product
        WHERE product_id = ANY(%s);
        """
        cur.execute(sql_query, (product_id_list,))
        products = cur.fetchall()  # List of (product_id, product_name)
        
        result = []
        
        for product_id, product_name in products:
            # Call the other API with product_name and user_email
            payload = {
                'product_name': product_name,
                'user_email': user_email
            }
            # Replace 'http://other-api.com/get_review_similarity' with the actual API endpoint
            response = requests.post('http://other-api.com/get_review_similarity', json=payload)
            
            if response.status_code == 200:
                data = response.json()  # Should be a list of tuples
                reviews_list = []
                for review_tuple in data:
                    # Each review_tuple is (review, similarity)
                    review, similarity = review_tuple
                    review_dict = {'review': review, 'profile_score': similarity}
                    reviews_list.append(review_dict)
                # Add to result
                product_info = {
                    'product_id': product_id,
                    'product_name': product_name,
                    'reviews': reviews_list
                }
                result.append(product_info)
            else:
                # Handle error, perhaps log it or append an empty reviews list
                product_info = {
                    'product_id': product_id,
                    'product_name': product_name,
                    'reviews': []
                }
                result.append(product_info)
        
        sorted_profiles = iterate(result)
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify(sorted_profiles), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def convert_textvec_to_profilejson(textvec):
    keys = [
        "Height", "Weight", "Body Mass Index (BMI)", "Skin Tone", "Hair Color", "Eye Color",
        "Extroversion", "Introversion", "Agreeableness", "Conscientiousness", "Neuroticism", 
        "Openness to Experience", "Intelligence Quotient", "Verbalprehension", "Perceptual Reasoning", 
        "Working", "Processing Speed", "Self-Awareness", "Self-Regulation", "Motivation", "Empathy", 
        "Social Skills", "Sociability", "Assertiveness", "Cooperativeness", "Conflict Resolution", 
        "Leadership Ability", "Age", "Education Level", "Occupation", "Marital Status", 
        "Number of Children", "Language Proficiency", "Musical Ability", "Artistic Ability", 
        "Athletic Ability", "Technical Skills", "Physical Health", "Mental Health", 
        "Chronic Conditions", "Disabilities", "Personality Type", "Learning Style", "Creativity", 
        "Adaptability", "Race", "Ethnicity"
    ]

    # Convert the input string to a list of integers
    values = list(map(int, textvec.split(", ")))

    # Create a dictionary by zipping the keys and values
    result = dict(zip(keys, values))
    
    return json.dumps(result, indent=2)

def get_product_id_by_name(product_name):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT product_id FROM products WHERE product_name = %s', (product_name,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()

        if result:
            return result[0]
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def get_user_id_by_email(user_email):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT id FROM users WHERE user_email = %s', (user_email,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()

        if result:
            return result[0]
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None
    
def get_embedding_by_id(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT 40-dim FROM users WHERE id = %s', (user_id,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()

        if result:
            return result[0]
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    app.run()