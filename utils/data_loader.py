import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sbn

def preprocess_data(df):
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=['Length1', 'Length2', 'Length3', 'Height', 'Width'])
    return df

def load_data(data_path):
    df = pd.read_csv(data_path)
    # sbn.pairplot(data=df, hue='Weight')
    
    X = df.drop(columns = ['Weight'])
    X = preprocess_data(X)
    y = df['Weight']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2003)

    return X_train, X_test, y_train, y_test 







