import nlpaug.augmenter.char as nac
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC


def augment_text(text):
    augmenter = nac.RandomCharAug(action="insert")

    augmented_text = augmenter.augment(text)

    return augmented_text


def preprocess_text(text):
    return text.lower().split()


def build_doc2vec_model(corpus):
    documents = [TaggedDocument(text, [i]) for i, text in enumerate(corpus)]
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
    return model


def svm_flow(df):
    labels = df['label'].values.astype("U")
    one_hot_encoded = LabelBinarizer().fit_transform(labels)
    X = df['sample'].values.astype("U")

    preprocessed_X = [preprocess_text(text) for text in list(X)]
    doc2vec_model = build_doc2vec_model(preprocessed_X)

    X_embeddings = [doc2vec_model.infer_vector(tokens) for tokens in preprocessed_X]
    X_train, X_test, Y_train, Y_test = train_test_split(X_embeddings, one_hot_encoded, test_size=0.1,
                                                        random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5,
                                                    random_state=42)

    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, np.argmax(Y_train, axis=1))

    # Predict on the test set
    y_pred = svm_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(np.argmax(Y_test, axis=1), y_pred)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
