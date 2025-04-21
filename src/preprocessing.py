import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path: str):
    """
    this loads and preprocesses the dataset.

    Parameters:
    - csv_path: str, path to the CSV file

    Returns:
    - X_train, X_test, y_train, y_test: numpy arrays
    """
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['class'])
    y = df['class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=0
    )
    
    return X_train, X_test, y_train, y_test
