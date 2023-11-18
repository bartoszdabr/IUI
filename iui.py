import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from tensorflow_text import BertTokenizer


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
    print(f'Acc: {accuracy}')


def build_bert_model():
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    bert_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_bert')

    #     # Bert Embedding
    preprocessed_text = bert_preprocess(bert_input)
    outputs = bert_encoder(preprocessed_text)
    pooled_output = tf.keras.layers.concatenate(
        tuple([outputs['encoder_outputs'][i] for i in range(-4, 0)]),
        name='last_4_hidden_states',
        axis=-1
    )[:, 0, :]
    pooled_output = tf.keras.layers.LayerNormalization()(pooled_output)

    print(pooled_output.shape)
    Dense = tf.keras.layers.Dropout(0.5)(pooled_output)
    Dense = tf.keras.layers.Dense(768, activation='relu')(Dense)
    Dense = tf.keras.layers.Dropout(0.5)(Dense)
    classifer = tf.keras.layers.Dense(1, activation='softmax', name="output")(Dense)
    model = tf.keras.models.Model(inputs=[bert_input], outputs=[classifer])

    return model


def train_test_bert(X_train, X_test, X_val, Y_train, Y_test, Y_val):
    check_point = tf.keras.callbacks.ModelCheckpoint(
        '/kaggle/working/check_point.hdf5',
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch"
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model = build_bert_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    h = model.fit(X_train, Y_train, epochs=20, validation_data=(X_val, Y_val), callbacks=[early_stopping, check_point])
    pd.DataFrame(h.history).plot(figsize=(14, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig('bert_count_vectorizer.png', dpi=300, figsize=(8, 6), transparent=True)
    plt.show()
    model.evaluate(X_test, Y_test)


df = pd.read_csv('dbdata.csv', encoding='utf-8')

stop_words_pl = get_stopwords()
vectorizer = CountVectorizer(stop_words=list(stop_words_pl))
X = vectorizer.fit_transform(df['sample'].values.astype("U"))

encoded_labels = LabelEncoder().fit_transform(df['label'].values.astype("U"))

X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_labels, test_size=0.1,
                                                    random_state=42)

train_test_bayes(X_train, X_test, Y_train, Y_test)

from sklearn.preprocessing import LabelBinarizer

labels = df['label'].values.astype("U")
one_hot_encoded = LabelBinarizer().fit_transform(labels)
X = df['sample'].values.astype("U")
text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000,
                                                    output_sequence_length=24,
                                                    standardize="lower_and_strip_punctuation",
                                                    split="whitespace",
                                                    output_mode="int")
text_vectorizer.adapt(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, one_hot_encoded, test_size=0.1,
                                                    random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5,
                                                random_state=42)
check_point = tf.keras.callbacks.ModelCheckpoint(
    '/kaggle/working/check_point.hdf5',
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch"
)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


robert_preprocess = hub.KerasLayer("https://kaggle.com/models/kaggle/roberta/frameworks/TensorFlow2/variations/en-cased-preprocess/versions/1")
robert_encoder = hub.KerasLayer("https://www.kaggle.com/models/kaggle/roberta/frameworks/TensorFlow2/variations/en-cased-l-12-h-768-a-12/versions/1", trainable=True)
input_Prompt = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input Prompt")

# BERT layers
Prompt = robert_preprocess(input_Prompt)
Prompt = robert_encoder(Prompt)["sequence_output"]

# LayerNormalization Layers
Prompt = tf.keras.layers.LayerNormalization(name="LayerNormalization_Promt")(Prompt[:, 0, :])

# Dropout layers
Prompt = tf.keras.layers.Dropout(0.3, name="Dropout_Prompt")(Prompt)

# Dense layers
Prompt = tf.keras.layers.Dense(256, activation="relu", name="RE_lu_dense_Prompt")(Prompt)

outputs = tf.keras.layers.Dense(8, activation="softmax", name="outputs")(Prompt)

model = tf.keras.models.Model(inputs=input_Prompt, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_val, Y_val),
              callbacks=[early_stopping, check_point])
model.save('model.h5')
accuracy = history.history["accuracy"]
loss = history.history["loss"]

# Getting the Validation accuracy and loss values
val_accuracy = history.history["val_accuracy"]
val_loss = history.history["val_loss"]

epochs = range(len(history.history["accuracy"]))

# Plot the loss and accuracy curves
plt.plot(epochs, accuracy, label="Training Accuracy")
plt.plot(epochs, loss, label="Training Loss")
plt.title("Training Accuracy and Loss Curves")
plt.xlabel("Epochs")
plt.legend()

# Plot the loss and accuracy curves of validation data
plt.figure()
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title("Validation Accuracy and Loss Curves")
plt.xlabel("Epochs")
plt.legend()
plt.savefig('bert_count_vectorizer.png')
model.evaluate(X_test, Y_test)
