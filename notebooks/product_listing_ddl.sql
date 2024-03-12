CREATE TABLE product_listing (
    id INT NOT NULL AUTO_INCREMENT  PRIMARY KEY,
    train_id INT,
    name VARCHAR(255),
    item_condition_id INT,
    category_name VARCHAR(255),
    brand_name VARCHAR(255),
    price FLOAT,
    shipping INT,
    item_description TEXT,
    created_at DATETIME,
    last_updated_at TINYINT
);


-- INSERT INTO product_listing (train_id, name, item_condition_id, category_name, brand_name, price, shipping, item_description, created_at, last_updated_at)
-- VALUES (1, 'Product Name', 1, 'Category', 'Brand', 10.99, 1, 'Description', NOW(), NOW());
