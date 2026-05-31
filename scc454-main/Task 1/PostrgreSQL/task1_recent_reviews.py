import time
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/amazon")

# Product Information Retrieval: Given a product ID (parent asin), retrieve
# the product title and full product description.
def product_information(product_id):
    query = f"""
    SELECT title, product_description
    FROM products
    WHERE product_id = '{product_id}'
    """
    start = time.time()
    df = pd.read_sql(query, engine)
    end = time.time()
    print(f"Retrieved {len(df)} product information for {product_id} in {(end-start)*1000:.2f} ms")
    return df
 
# Recent Reviews: Given a product ID (parent asin), retrieve the most recent
# N customer reviews, including rating, review title, review text, and timestamp.   

def recent_reviews(asin, n_reviews):
    query = f"""
    SELECT rating, review_title, review_text, review_time
    FROM reviews
    WHERE product_id = '{asin}'
    ORDER BY review_time DESC
    LIMIT {n_reviews};
    """
    
    start = time.time()
    df = pd.read_sql(query, engine)
    end = time.time()
    
    print(f"Retrieved {len(df)} recent reviews for {asin} in {(end-start)*1000:.2f} ms")
    return df

# Keyword Search: Given a keyword, return all products 
# whose title or description contains the keyword.
def keyword_search(keyword):

    query = f"""
    SELECT product_id, title, product_description
    FROM products
    WHERE LOWER(title) LIKE '%%{keyword.lower()}%%'
    OR LOWER(product_description) LIKE '%%{keyword.lower()}%%';
    """
    
    start_time = time.time()
    df = pd.read_sql(query, engine)
    end_time = time.time()
    
    duration_ms = (end_time - start_time) * 1000
    print(f"Keyword '{keyword}' found {len(df)} products in {duration_ms:.2f} ms")
    return df

def indexed_keyword_search(keyword):
    # Use the @@ operator to trigger the GIN index
    query = f"""
    SELECT product_id, title
    FROM products
    WHERE search_vector @@ plainto_tsquery('english', '{keyword}');
    """
    
    start = time.time()
    df = pd.read_sql(query, engine)
    end = time.time()
    
    print(f"Indexed search for '{keyword}' took {(end-start)*1000:.2f} ms")
    return df

# User Review History: Given a user ID, retrieve all products reviewed by that
# user along with the corresponding ratings and review dates.

def review_history(user_id):
    query = f"""
    SELECT product_id, rating, review_time
    FROM reviews
    WHERE user_id = '{user_id}'
    """
    
    start = time.time()
    df = pd.read_sql(query, engine)
    end = time.time()
    
    print(f"Retrieved {len(df)} product reviewed by {user_id} in {(end-start)*1000:.2f} ms")
    return df

# Product Statistics: Retrieve the average rating, rating distribution 
# (count per star level), and total number of reviews for a given product
def product_stat(product_id):
    query = f"""
    SELECT  
        AVG(rating) OVER() as average_rating,
        COUNT(*) as rating_distribution,
        SUM(COUNT(*)) OVER() as total_reviews
    FROM reviews
    WHERE product_id = '{product_id}'
    GROUP BY rating;
    """
    
    start = time.time()
    df = pd.read_sql(query, engine)
    end = time.time()
    
    print(f"Retrieved product stat in {(end-start)*1000:.2f} ms")
    return df
    

# evaluate requirement 1 -> product information
product_id = 'B07FVZVQKV'
print("Evaluate requirement 1st")
print(product_information(product_id))

# evaluate requirement 2 -> recent reviews
target_asin = 'B07G9GWFSM'
print("Evaluate requirement 2nd")
print(recent_reviews(target_asin, 5))

# evaluate requirement 3 -> keyword search
print("Evaluate requirement 3rd (without index)")
print(keyword_search("shampoo"))

# evaluate requirement 3 -> keyword search
print("Evaluate requirement 3rd (with index)")
print(indexed_keyword_search("shampoo"))

# evaluate requirement 4 -> user review history
user_id = 'AFLX66DKF6R3H6OEOC3TIVAYXZIQ'
print("Evaluate requirement 4th")
print(review_history(user_id))

# evaluate requirement 5 -> product stat
product_id = 'B07FVZVQKV'
print("Evaluate requirement 5th")
print(product_stat(product_id))