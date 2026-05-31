CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(20),
    user_id VARCHAR(50),
    rating NUMERIC(3,2),
    review_title TEXT,
    review_text TEXT,
    review_time TIMESTAMP,
    verified_purchase INTEGER,
    helpful_vote SMALLINT
);