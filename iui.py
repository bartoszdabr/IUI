import argparse
import pandas as pd

from bayes import bayes_flow
from roberta import roberta_flow


def parse_args() -> str:
    parser = argparse.ArgumentParser(description="model type")
    parser.add_argument('--model', choices=['bayes', 'roberta'], default='bayes',
                        help='Choose the model to train (default is Bayes)')
    parser.add_argument('--language', choices=['eng', 'pl'], default='pl',
                        help='Choose dataset language to train and validate model (default pl)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_type = args.model
    language = args.language
    if language == 'pl':
        df = pd.read_csv('dbdata.csv', encoding='utf-8')
    elif language == 'eng':
        df = pd.read_csv('dbdata_eng.csv', encoding='utf-8')
    print(f'Training {model_type} | language: {language}')
    if model_type == 'bayes':
        bayes_flow(df)
    elif model_type == 'roberta':
        roberta_flow(df)
