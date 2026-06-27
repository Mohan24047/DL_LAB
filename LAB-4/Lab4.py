import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training Images:", x_train.shape)
print("Training Labels:", y_train.shape)
print("Testing Images :", x_test.shape)
print("Testing Labels :", y_test.shape)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

plt.figure(figsize=(10,4))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")

plt.show()

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )

])

model.summary()

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

history = model.fit(

    x_train,
    y_train,

    validation_split=0.2,

    epochs=5,

    batch_size=32,

    verbose=1

)

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.grid()

plt.show()

loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss :", loss)
print("Test Accuracy :", accuracy)

predictions = model.predict(x_test)

predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(
    y_test,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

fig, ax = plt.subplots(figsize=(10,10))

disp.plot(ax=ax)

plt.title("Confusion Matrix")

plt.show()

print(classification_report(
    y_test,
    predicted_labels
))

misclassified = np.where(predicted_labels != y_test)[0]

plt.figure(figsize=(12,4))

for i in range(3):

    index = misclassified[i]

    plt.subplot(1,3,i+1)

    plt.imshow(x_test[index], cmap="gray")

    plt.title(
        f"True: {y_test[index]}\nPred: {predicted_labels[index]}"
    )

    plt.axis("off")

plt.show()

