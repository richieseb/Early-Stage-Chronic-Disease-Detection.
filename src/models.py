from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def train_probabilistic_model(X_train, y_train):
    """Initializes and fits a baseline Gaussian Naive Bayes classifier."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_optimized_knn(X_train, y_train, cv_folds=5):
    """
    Sweeps neighborhood options using Cross-Validation to prevent overfitting,
    returning an optimized KNN framework.
    """
    best_k = 3
    best_score = 0
    
    # Evaluate odd neighbors to prevent tie-breaking deadlock votes
    for k in [3, 5, 7, 9]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv_folds, scoring='accuracy')
        mean_score = np.mean(scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
            
    print(f"[INFO] Hyperparameter Optimization complete. Optimal K Selected: {best_k}")
    optimized_knn = KNeighborsClassifier(n_neighbors=best_k)
    optimized_knn.fit(X_train, y_train)
    return optimized_knn, best_k
