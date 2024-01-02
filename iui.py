import argparse
import pandas as pd

from bayes import bayes_flow
from roberta import roberta_flow


def get_model() -> str:
    parser = argparse.ArgumentParser(description="model type")
    parser.add_argument('--model', choices=['bayes', 'roberta'], default='bayes',
                        help='Choose the model to train (default is Bayes)')
    args = parser.parse_args()
    return args.model


if __name__ == "__main__":
    df = pd.read_csv('dbdata.csv', encoding='utf-8')
    model_type = get_model()
    print(f'Training {model_type}')
    if model_type == 'bayes':
        bayes_flow(df)
    elif model_type == 'roberta':
        roberta_flow(df)
