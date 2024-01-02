import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder


def get_stopwords():
    stop_words_pl = set(pd.read_csv('stopwords.txt')['a'].values)
    klient = open("klient.csv", "r")
    odmiany = klient.readlines()

    for odmiana in odmiany:
        lament = odmiana.split('\n')
        odmiana = lament[0]
        stop_words_pl.add(odmiana)

    return stop_words_pl


def train_test_bayes(X_train, X_test, Y_train, Y_test):
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    f_score = f1_score(Y_test, y_pred, average=None)
    print(f'Bayes acc: {accuracy} \n f1_score: {f_score}')


def bayes_flow(df) -> None:
    stop_words_pl = get_stopwords()
    vectorizer = CountVectorizer(stop_words=list(stop_words_pl))
    X = vectorizer.fit_transform(df['sample'].values.astype("U"))
    encoded_labels = LabelEncoder().fit_transform(df['label'].values.astype("U"))
    X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_labels, test_size=0.1,
                                                        random_state=42)
    train_test_bayes(X_train, X_test, Y_train, Y_test)
