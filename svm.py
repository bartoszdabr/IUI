import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


stopwords_file1 = "stopwords.txt"
stopwords_file2 = "klient.csv"

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().splitlines()
    return set(stopwords)

stopwords1 = load_stopwords(stopwords_file1)
stopwords2 = load_stopwords(stopwords_file2)


def preprocess_text(text):
    return text.lower().split()


def build_doc2vec_model(corpus, tags, vector_size=50, min_count=2, epochs=30):
    documents = [TaggedDocument(words=text, tags=[tags[i]]) for i, text in enumerate(corpus)]
    model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)

    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    return model


def svm_flow(df):
    # Prepare data
    labels = df['label'].values.astype("U")
    label_encoder = LabelEncoder()
    one_hot_encoded = label_encoder.fit_transform(labels)
    X = df['sample'].values.astype("U")

    preprocessed_X = [preprocess_text(text) for text in list(X)]

    # Hyperparameter tuning for Doc2Vec
    for vector_size in [50, 100]:
        for min_count in [2, 5]:
            for epochs in [30, 50]:
                print("---------")
                print(f"Training Doc2Vec with vector_size={vector_size}, min_count={min_count}, epochs={epochs}")

                doc2vec_model = build_doc2vec_model(preprocessed_X, labels, vector_size=vector_size,
                                                    min_count=min_count, epochs=epochs)

                X_embeddings = [doc2vec_model.infer_vector(tokens) for tokens in preprocessed_X]
                X_train, X_test, Y_train, Y_test = train_test_split(X_embeddings, one_hot_encoded, test_size=0.05,
                                                                    random_state=42)

                # Hyperparameter tuning for SVM
                for kernel in ['linear']:
                    print(f"Training SVM with kernel: {kernel}")
                    svm_model = SVC(kernel=kernel)
                    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 'scale', 'auto']}
                    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    grid_search.fit(X_train, Y_train)
                    print("Best Parameters:", grid_search.best_params_)
                    best_svm_model = grid_search.best_estimator_

                    # Calculate cross validation
                    cv_scores = cross_val_score(best_svm_model, X_train, Y_train, cv=5, scoring='accuracy', n_jobs=-1)
                    print("Cross-Validation Scores:", cv_scores)
                    print("Mean Accuracy:", np.mean(cv_scores))

                    # Predict on the test set
                    y_pred = best_svm_model.predict(X_test)

                    # Calculate accuracy
                    accuracy = accuracy_score(Y_test, y_pred)
                    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

                    # Find top 10 most important words for each class
                    class_labels = label_encoder.classes_
                    svm_coef = best_svm_model.coef_
                    for i, class_label in enumerate(class_labels):
                        class_idx = label_encoder.transform([class_label])[0]
                        class_coef = svm_coef[class_idx]
                        top_indices = np.argsort(class_coef)[::-1]
                        top_words = [doc2vec_model.wv.index_to_key[idx] for idx in top_indices]
                        top_words = [word for word in top_words if word not in stopwords1 and word not in stopwords2]
                        top_words = top_words[:20]
                        print(f"Top words for class '{class_label}': {', '.join(top_words)}")
                    print("---------")
