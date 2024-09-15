import logging
from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv
import psycopg2
import json

load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
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
    app.logger.info("add_user route accessed")
    # ... rest of the function ...

# Add this at the end of your file
if __name__ == "__main__":
    app.run(debug=True)
    
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
            'INSERT INTO Users (id, email, profile) VALUES (%s, %s, %s)',
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
            SELECT user_review.user_id, Reviews.text
            FROM user_review
            JOIN Reviews ON user_review.review_id = Reviews.id
            WHERE user_review.user_id IN ({format_strings});
        """, tuple(user_ids))

        review_data = cur.fetchall()

        cur.close()
        conn.close()

        user_to_text = {user_id: text for user_id, text in review_data}

        result = [(user_to_text.get(user_id, None), int_val) for user_id, int_val in sorted_list_of_user_correlation]

        return jsonify({'message': result}), 201

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
    
if __name__ == "__main__":
    app.run()