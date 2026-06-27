import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=np.float32)

y = np.array([[0],[1],[1],[0]], dtype=np.float32)


def train_model(activation='sigmoid', lr=0.01, epochs=100):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation=activation, input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X,
        y,
        epochs=epochs,
        verbose=0
    )

    loss, accuracy = model.evaluate(X, y, verbose=0)

    print(f"\nActivation = {activation}")
    print(f"Learning Rate = {lr}")
    print(f"Final Loss = {loss:.4f}")
    print(f"Final Accuracy = {accuracy*100:.1f}%")

    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'])
    plt.title(f'{activation.upper()}  LR={lr}')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return history


train_model('sigmoid',0.01)
train_model('sigmoid',0.1)
train_model('relu',0.01)
train_model('relu',0.1)