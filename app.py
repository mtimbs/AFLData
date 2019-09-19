from Models.knn import knn

if __name__ == "__main__":
    knn_model, metrics = knn(neighbours=10, sample_probability=10)
    print(metrics)
