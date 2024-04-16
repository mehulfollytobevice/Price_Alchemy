from sklearn.linear_model import LinearRegression,HuberRegressor, RANSACRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

models={
    "huber": HuberRegressor(), 
    "huber_modified": HuberRegressor(epsilon=1.35,max_iter=200,alpha=0.01),
    "knn": KNeighborsRegressor(n_neighbors=10, weights='distance'),
    "ransac": RANSACRegressor(),
    "mlp": MLPRegressor(hidden_layer_sizes=(100,100),learning_rate='adaptive',
                        learning_rate_init=0.01, early_stopping=True,
                        random_state=42, max_iter=100),
    "mlp_three":MLPRegressor(hidden_layer_sizes=(100,100,100),learning_rate='adaptive',
                        learning_rate_init=0.01, early_stopping=True,
                        random_state=42, max_iter=100)
}