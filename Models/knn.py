from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split


def knn(feature_training_data, target_training_data, neighbours):
    # instantiate learning model (k = 3)
    knn = KNeighborsClassifier(n_neighbors=neighbours)
    # fitting the model
    knn.fit(feature_training_data, target_training_data)

    return knn
