import tensorflow as tf
import numpy as np
import pandas as pd

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=np.float32)

y = np.array([[0],[1],[1],[0]], dtype=np.float32)

trials = [
    {"lr":0.01,"hidden":4},
    {"lr":0.10,"hidden":4},
    {"lr":0.01,"hidden":8},
    {"lr":0.10,"hidden":8},
    {"lr":0.05,"hidden":6}
]

results = []

for trial in trials:

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            trial["hidden"],
            activation='sigmoid',
            input_shape=(2,)
        ),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=trial["lr"]
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X,
        y,
        epochs=100,
        verbose=0
    )

    loss, acc = model.evaluate(X, y, verbose=0)

    results.append([
        trial["lr"],
        trial["hidden"],
        loss,
        acc
    ])

df = pd.DataFrame(
    results,
    columns=[
        "Learning Rate",
        "Hidden Units",
        "Training Loss",
        "Accuracy"
    ]
)

print(df)