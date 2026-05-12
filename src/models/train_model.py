import pickle
from pathlib import Path
from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.predict_model import build_recommender, MODES


def train(raw_dir='data/raw', model_dir='models'):
    smd = make_dataset(raw_dir)
    smd = build_features(smd)


    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for mode in MODES:
        print(f"Training {mode} model...")
        rec = build_recommender(smd, mode=mode)
        path = model_dir / f'recommender_{mode}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(rec, f)
        print(f"  Saved to {path}")

    return rec


if __name__ == '__main__':
    train()
