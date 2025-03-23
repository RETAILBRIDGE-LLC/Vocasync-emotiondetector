import argparse
import pandas as pd
import xgboost as xgb
import os

def train(train_path, model_output_path):
    df = pd.read_csv(train_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    dtrain = xgb.DMatrix(X, label=y)

    model = xgb.train(params={"objective": "binary:logistic"}, dtrain=dtrain, num_boost_round=100)
    os.makedirs(model_output_path, exist_ok=True)
    model.save_model(f"{model_output_path}/xgboost-model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train/train.csv')
    parser.add_argument('--model_output', type=str, default='/opt/ml/model')
    args = parser.parse_args()
    train(args.train, args.model_output)
