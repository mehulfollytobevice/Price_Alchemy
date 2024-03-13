CREATE TABLE holdout_listing (
    id INT NOT NULL AUTO_INCREMENT  PRIMARY KEY,
    train_id INT,
    name VARCHAR(255),
    item_condition_id INT,
    category_name VARCHAR(255),
    brand_name VARCHAR(255),
    price FLOAT,
    shipping INT,
    item_description TEXT,
    is_migrated TINYINT DEFAULT 0
);