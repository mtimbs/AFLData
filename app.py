from sklearn.model_selection import train_test_split

from Models.knn import knn
from Helpers.visualisations import plot_confusion_matrix
from PreProcessing.load_data import all_data_without_contested_possession_breakdown


def knn_baseline():
    '''
    Basic implementation of KNN algorithm to get a baseline confusion matrix for brownlow predictions.
    '''
    features, targets = all_data_without_contested_possession_breakdown()

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=0)

    knn_model = knn(X_train, y_train, 10)

    predictions = knn_model.predict(X_test)

    plot_confusion_matrix(y_test, predictions, 'KNN - No sampling, 10 Neighbours')


if __name__ == "__main__":
    knn_baseline()
