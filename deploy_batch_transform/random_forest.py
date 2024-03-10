# random_forest.py
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker固有の引数
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()

    # トレーニングデータを読み込み
    train_data = pd.read_csv(os.path.join(args.train, 'iris_train.csv'), header=0)
    
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']

    # モデルをトレーニング
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # モデルを保存
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

def model_fn(model_dir):
    """モデルファイルをロードするための関数。
    
    Args:
        model_dir (str): モデルファイルが保存されているディレクトリのパス。
    
    Returns:
        model: ロードされたモデルオブジェクト。
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model