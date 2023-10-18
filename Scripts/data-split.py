import pandas as pd
from sklearn.model_selection import train_test_split

def SplitData():
    data = pd.read_csv("Data/data.csv")
    train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['Target'])
    train, val = train_test_split(train, test_size=0.125, shuffle=True, stratify=train['Target'])
    
    train.to_csv("Data/train.csv")
    test.to_csv("Data/test.csv")
    val.to_csv("Data/val.csv")
    
if __name__ == '__main__':
    SplitData()
