from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.extract_features import get_data_from_workbook
# from sklearn.model_selection import train_test_split
from openpyxl import load_workbook
from src.utils import load_config
import argparse
from src.data import FeatureExtractor
from src.data import get_label_map
from src.utils import load_labels
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--config")
    parser.add_argument("--cat_ids", help="path to labels.txt file")
    args = parser.parse_args()

    if "csv" in args.data:
        df = pd.read_csv(args.data)
    elif "xls" in args.data:
        config = load_config(args.config, type=None)
        data = get_data_from_workbook(args.data, **config.wb.to_dict())
        df = pd.DataFrame(data)

    train_e2e(df, args.cat_ids)


def train_e2e(df, cat_ids):
    df.dropna(axis="index", inplace=True)
    X, Y = preprocess(df, cat_ids)

    ft_ext = FeatureExtractor()
    clf = RandomForestClassifier()
    pipe = Pipeline([("feature_extractor", ft_ext), ("model", clf)])

    param_grid = {"model__n_estimators": range(20, 200, 20)}

    gs = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, verbose=1)
    gs.fit(X, Y)

    print("best_score", gs.best_score_)

    model = gs.best_estimator_
    dump(model, 'src/model.joblib')


def preprocess(df, labels, val_split=0.2):

    Y = df.pop("cat").str.lower().str.strip()
    mp = get_label_map(load_labels(labels))
    Y = Y.apply(lambda x: map_ids(x, mp))
    X = df

    return X, Y


def map_ids(x, mp):
    if x in mp:
        return mp[x]
    else:
        print(f"Could not find {x} in labels.txt, exiting!")
        exit(1)


if __name__ == "__main__":
    main()
