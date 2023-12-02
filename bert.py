import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt


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
