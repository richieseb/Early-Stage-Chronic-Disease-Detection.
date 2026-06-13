import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import clean_and_scale_data
from models import train_probabilistic_model, train_optimized_knn

def generate_synthetic_clinical_data(samples=600):
    """Generates standard baseline clinical data vectors for structural tracking."""
    np.random.seed(42)
    bp = np.random.normal(75, 12, samples)
    creatinine = np.random.exponential(1.2, samples)
    
    df = pd.DataFrame({'Blood_Pressure': bp, 'Serum_Creatinine': creatinine})
    risk = (df['Serum_Creatinine'] * 1.8) + (np.abs(df['Blood_Pressure'] - 75) * 0.05)
    df['Diagnosis_Target'] = (risk > 2.2).astype(int)
    return df

if __name__ == "__main__":
    # Ingest clinical samples
    raw_data = generate_synthetic_clinical_data()
    
    # Process attributes through our pipeline
    X, y, data_scaler = clean_and_scale_data(raw_data)
    
    # Partition into independent evaluation spaces
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train classifiers
    gnb = train_probabilistic_model(X_train, y_train)
    knn, optimal_k = train_optimized_knn(X_train, y_train)
    
    # Generate evaluation prints
    print("\n" + "="*50)
    print("         ACADEMIC PERFORMANCE SUMMARY")
    print("="*50)
    
    for name, model in [("Gaussian Naive Bayes", gnb), (f"KNN (K={optimal_k})", knn)]:
        preds = model.predict(X_test)
        print(f"\nModel Frame: {name}")
        print("-" * 30)
        print(classification_report(y_test, preds, target_names=['Healthy', 'At Risk']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
