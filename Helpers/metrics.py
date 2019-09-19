from sklearn.metrics import confusion_matrix, classification_report


def metrics(true, predicted):
    return classification_report(true, predicted)
