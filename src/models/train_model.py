import pickle
from pathlib import Path
from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.predict_model import build_recommender


def train(raw_dir='data/raw', model_dir='models'):
    smd = make_dataset(raw_dir)

    smd = build_features(smd)

    recommender = build_recommender(smd)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / 'recommender.pkl', 'wb') as f:
        pickle.dump(recommender, f)
    print(f"Model saved to {model_dir / 'recommender.pkl'}")

    return recommender


if __name__ == '__main__':
    train()
