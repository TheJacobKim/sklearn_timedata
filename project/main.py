print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

f = open('result.txt','w')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

scores = ['accuracy', 'precision', 'recall']

classifiers = [
    KNeighborsClassifier(),
    SVC(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier()
    # GaussianProcessClassifier(),future purposes
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
]

# Set the parameters by cross-validation
k_range = list(range(1, 31))
n_range = list(range(1, 10))
m_range = list(range(1, 3))
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
tuned_parameters = [
    {'n_neighbors': k_range},  # knn
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 20, 100, 1000]},  # RBF SVM
    {'kernel': ['linear'], 'C': [1, 20, 100, 1000]},  # Linear SVM"
    {'max_depth': n_range},  # Decision Tree
    {'max_depth': n_range, 'n_estimators': n_range, 'max_features': m_range},  # Random Forest
    {'alpha': m_range},  # Neural Net
    # {'kernel': kernel},  # Gaussian Process
]

# Create random dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=1),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

dataset_num = 0
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    f.write("\n\n\nRandom data set #" + str(dataset_num))
    dataset_num += 1
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    # iterate over classifiers
    i = 0
    for name, clf in zip(names, classifiers):
        grid = GridSearchCV(clf, tuned_parameters[i], cv=10, scoring='accuracy')
        i += 1
        grid.fit(X_train, y_train)

        # Best estimator overall
        f.write("\n\n____________________________________________________________\n")
        f.write("Best estimator\n")
        f.write(str(grid.best_estimator_))

        # What was the best param?
        f.write("\n\nBest param\n")
        f.write(str(grid.best_params_))

        # Save params & mean fit time and compare
        param_list = grid.cv_results_['params']
        mean_fit_time = grid.cv_results_['mean_fit_time']
        f.write("\n\nMean fit time\n")
        for param, time in zip(param_list, mean_fit_time):
            f.write(str(param) + ":  ")
            f.write(str(time) + '\n')


f.close()