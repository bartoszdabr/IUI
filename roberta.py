import datetime
import os.path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorboard.plugins.hparams import api as hp

HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))


def build_roberta(hparams):
    robert_preprocess = hub.KerasLayer(
        "https://kaggle.com/models/kaggle/roberta/frameworks/TensorFlow2/variations/en-cased-preprocess/versions/1")
    robert_encoder = hub.KerasLayer(
        "https://www.kaggle.com/models/kaggle/roberta/frameworks/TensorFlow2/variations/en-cased-l-24-h-1024-a-16/versions/1",
        trainable=True)
    input_prompt = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input prompt")
    prompt = robert_preprocess(input_prompt)
    prompt = robert_encoder(prompt)["sequence_output"]

    # LayerNormalization Layers
    prompt = tf.keras.layers.LayerNormalization(name="LayerNormalization_Promt")(prompt[:, 0, :])
    # Dropout layers
    prompt = tf.keras.layers.Dropout(hparams[HP_DROPOUT], name="Dropout_Prompt")(prompt)
    # Dense layers
    prompt = tf.keras.layers.Dense(256, activation="relu", name="RE_lu_dense_Prompt")(prompt)
    outputs = tf.keras.layers.Dense(8, activation="softmax", name="outputs")(prompt)
    model = tf.keras.models.Model(inputs=input_prompt, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def test_train(X_train, Y_train, X_val, Y_val, X_test, Y_test, hparams) -> None:
    check_point_path = os.path.join("checkpoints/roberta/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                    "check_point.hdf5")
    check_point = tf.keras.callbacks.ModelCheckpoint(
        check_point_path,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch"
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    log_dir = os.path.join("logs/roberta/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    hparams_callback = hp.KerasCallback(log_dir, hparams)
    import tensorflow_model_optimization as tfmot
    model = build_roberta(hparams)
    model = tfmot.quantization.keras.quantize_model(original_model)
    model.fit(X_train, Y_train, epochs=40, batch_size=16, validation_data=(X_val, Y_val),
              callbacks=[early_stopping, tensorboard_callback, hparams_callback])

    print(model.evaluate(X_test, Y_test, return_dict=True))
    # fscore cannot be calculated as a callback due to bug in tensorflow
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    f_score = f1_score(np.argmax(Y_test, axis=1), y_pred, average=None)
    print(f'f_score={f_score}')


def roberta_flow(df) -> None:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
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

    for dropout_rate in range(1, 6, 1):
        hparams = {
            HP_DROPOUT: dropout_rate * 0.01,
        }
        print({h.name: hparams[h] for h in hparams})
        test_train(X_train, Y_train, X_val, Y_val, X_test, Y_test, hparams)
