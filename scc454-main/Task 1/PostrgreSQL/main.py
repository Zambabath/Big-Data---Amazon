import pandas as pd
from sqlalchemy import create_engine

# setup Connection
# be careful about posting password
engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/amazon")

# upload product data
meta_path = "/Users/prim/Desktop/MSc/454 Big data/Project/data/cleaned_meta_beauty.json"

# We use a chunksize of 10,000
for chunk in pd.read_json(meta_path, lines=True, chunksize=10000):
    products_df = chunk[['parent_asin', 'title', 'description','average_rating','main_category','rating_number','Age Range (Description)','Brand','Hair Type','Item Form','Material','features_string']].rename(
        columns={
            'parent_asin': 'product_id',
            'description': 'product_description',
            'Age Range (Description)': 'customer_age',
            'Brand':'brand',
            'Hair Type':'hair_type',
            'Item Form':'item_form',
            'Material':'material',
            'features_string':'feature'
        }
    )
    # insert data into a table 
    products_df.to_sql("products", engine, if_exists="append", index=False, method="multi", chunksize=1000)

print("Products uploaded!")

# upload Reviews
reviews_path = "/Users/prim/Desktop/MSc/454 Big data/Project/data/cleaned_review.json"

for chunk in pd.read_json(reviews_path, lines=True, chunksize=10000):
    review_df = chunk[['parent_asin', 'user_id','rating','title','text','timestamp','helpful_vote','verified_purchase']].rename(
        columns={
            'parent_asin': 'product_id',
            'title':'review_title',
            'text':'review_text',
            'timestamp':'review_time'
        }
    )
    
    review_df.to_sql("reviews", engine, if_exists="append", index=False, method="multi", chunksize=1000)

print("Reviews uploaded successfully!")

# create index
from sqlalchemy import text

def apply_indexing(engine):
    with engine.connect() as conn:
        print("Starting indexing process...")
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_reviews_product_id ON reviews(product_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_reviews_product_time ON reviews(product_id, review_time DESC);"))
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS search_vector tsvector;"))
        

        print("Populating search vectors (this may take a minute)...")
        conn.execute(text("""
            UPDATE products 
            SET search_vector = to_tsvector('english', coalesce(title, '') || ' ' || coalesce(product_description, ''));
        """))
        
        # Create the GIN Index
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_search ON products USING GIN(search_vector);"))
        
        # Commit changes to the database
        conn.commit()
        print("Indexing completed successfully!")

apply_indexing(engine)