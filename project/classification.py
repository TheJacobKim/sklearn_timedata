print(__doc__)

import warnings
warnings.simplefilter(action='ignore', category=Warning)
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import *
from sklearn.neural_network import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing

file = open('result_clf.csv', 'w')

ALGORITHM_NAMES = [#"kNN", "SVM", "Decision Tree",
         #"Random Forest", #"Neural Net", 
         "AdaBoost",
        ]

scores = ['accuracy', 'precision', 'recall']

classifiers = [
    #KNeighborsClassifier(),
    #SVC(),
    #DecisionTreeClassifier(),
    #RandomForestClassifier(),
    #MLPClassifier(),
    AdaBoostClassifier(),
]

regressors = [
    KNeighborsRegressor(),
    SVR(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    MLPRegressor(),
    AdaBoostRegressor(),
]

# Parameters used for parameter fields below
k_range = list(range(1, 10))
leaf_range = list(range(25, 35))
degree_range = list(range(2, 5))
n_range = list(range(1, 10))
m_range = list(range(1, 3))
hidden_layer_size_range = list(range(80, 150))

classifiers_list = [DecisionTreeClassifier(random_state=3),
                    DecisionTreeClassifier(random_state=4),
                    DecisionTreeClassifier(random_state=5),
                    DecisionTreeClassifier(random_state=6),
                    DecisionTreeClassifier(random_state=7),
                    DecisionTreeClassifier(random_state=8),
                    DecisionTreeClassifier(random_state=9),
                    DecisionTreeClassifier(random_state=10),
                    DecisionTreeClassifier(random_state=11),
                    ]

n_estimator_range = list(range(50, 200))

tuned_parameters_classifiers = [
    # knn classifier
    #{'n_neighbors': k_range, 'weights': ['uniform', 'distance'],
    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': leaf_range},

    # SVM classifier
    #{'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 20, 100, 1000],
    # 'degree': degree_range},

    # Decision Tree Classifier
    #{'max_depth': n_range, 'max_features': ['auto', 'sqrt', 'log2']},

    # Random Forest Classifier
    #{'max_depth': n_range, 'n_estimators': n_estimator_range, 'min_samples_leaf':[1,2,3], 'max_features': ['auto', 'sqrt', 'log2']},

    # Neural Net, MLPClassifier
    #{'hidden_layer_sizes': hidden_layer_size_range, 'activation': ['identity', 'logistic', 'tanh', 'relu'],
    #'solver': ['lbfgs', 'sgd', 'adam'], 'learning_rate': ['constant', 'invscaling', 'adaptive']},

    # AdaBoostClassifier
    {'base_estimator': classifiers_list, 'n_estimators': n_estimator_range},
]

# Create random dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

classification_datasets_names = ['make_moons', 'make_circles', 'linearly_separable', 'iris', 'digits', 'wine']
classification_datasets = [make_moons(n_samples=10000, noise=0.3, random_state=1),
                           make_circles(n_samples=10000, noise=0.2, factor=0.5, random_state=1),
                           linearly_separable,
                           load_iris(return_X_y = True),
                           load_digits(return_X_y = True),
                           load_wine(return_X_y = True)
                           ]

# iterate over datasets
dataset_num = 0
for ds_cnt, ds in enumerate(classification_datasets):
    name = classification_datasets_names[dataset_num]
    dataset_num  += 1
    print("Working on " + name + "dataset")
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=420)
    X_train = preprocessing.scale(X_train)

    i = 0
    # iterate over classifiers
    for clf in classifiers:
        try:
            print("Working on ",  str(clf))
            grid = GridSearchCV(clf, tuned_parameters_classifiers[i], cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)

            # CSV writer
            param_list = grid.cv_results_['params']
            fieldnames = list(param_list[0].keys())
            fieldnames.append('mean_fit_time')
            fieldnames.insert(0, 'algorithm_name')
            fieldnames.insert(1, 'dataset_name')
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            # Save params & mean fit time and compare
            mean_fit_time = grid.cv_results_['mean_fit_time']
            for param, time in zip(param_list, mean_fit_time):
                param['mean_fit_time'] = time
                param['algorithm_name'] = ALGORITHM_NAMES[i]
                param['dataset_name'] = name
                writer.writerow(param)

        except Exception as e:
           # pass
            print(e)
        i += 1

file.close()

