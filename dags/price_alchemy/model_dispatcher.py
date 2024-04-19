from sklearn.linear_model import LinearRegression,HuberRegressor, RANSACRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from hyperopt import hp,fmin,tpe,Trials
from hyperopt.pyll.base import scope

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

PARAMS={
    'hidden_layers':scope.int( hp.quniform('hidden_layers', 1, 5, 1)),  # Number of hidden layers: 1 to 5
    'hidden_neurons': scope.int(hp.quniform('hidden_neurons', 5, 100, 1)), 
    "max_iter":scope.int(hp.quniform("max_iter",100,200,1)),
    "learning_rate_init": hp.uniform("learning_rate_init",0.01,0.1),
    "batch_size":scope.int(hp.randint("batch_size",100,500)),
    "learning_rate":hp.choice("learning_rate",["invscaling","adaptive"])
}