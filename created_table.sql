CREATE TABLE product_listing (
    id INT AUTO_INCREMENT PRIMARY KEY,
    train_id INT,
    name VARCHAR(255),
    item_condition_id INT,
    category_name VARCHAR(255),
    brand_name VARCHAR(255),
    price FLOAT,
    shipping INT,
    item_description TEXT,
    created_at DATETIME,
    last_updated_at DATETIME
);
