from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from PreProcessing.load_data import all_data_without_contested_possession_breakdown
from Helpers.visualisations import plot_confusion_matrix


def knn(neighbours, sample_probability=None):
    """
        Basic implementation of KNN algorithm to get a baseline confusion matrix for brownlow predictions.
    """
    # Load the raw training data
    features, targets = all_data_without_contested_possession_breakdown(sample_probability)

    # Utilise sklearn's helper to split data into training/validation sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=0)

    # Initialise the KNN Classifier Model
    model = KNeighborsClassifier(n_neighbors=neighbours)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Get Predictions for the validation set
    predictions = model.predict(X_test)

    # Generate Confusion matrix for trained model based on validation performance
    title = f'KNN - {"No sampling" if sample_probability is None else "Sampling"}, {neighbours} Neighbours'
    plot_confusion_matrix(y_test, predictions, title)

    return model, classification_report(y_test, predictions)
