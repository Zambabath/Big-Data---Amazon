CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_id VARCHAR(10),
    title TEXT,
    product_description TEXT,
    rating_number DOUBLE,
    average_rating NUMERIC(3,2),
    main_category TEXT,
    customer_age TEXT,
    brand TEXT,
    hair_type TEXT,
    item_form TEXT,
    material TEXT,
    feature TEXT
);