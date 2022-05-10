from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = "data"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our data directory, then initialize the image class objects
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = DIRECTORY + "/" + str(category)

    for img in os.listdir(path):
        image_path = path + "/" + str(img)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)


# Hot-One Binary Encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# load the mobilenetv2 network, ensuring the head FC layers are kept
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model) # Just use relu
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model) # sigmoid would also work

# get the FC model on top of the base model -> this is the model we will train
model = Model(inputs=base_model.input, outputs=head_model)

# loop over all layers and freeze them so they will not be updated during the first training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head network
print("[INFO] training head...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
            steps_per_epoch=len(trainX) // BS,
            validation_data=(testX, testY),
            validation_steps=len(testX) // BS,
            epochs=EPOCHS
        )

# make predictions on the test set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)


# for each image in the test set we will find the index of the label with corresponding largest
# predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# save the Model
print("[INFO] saving model...")
model.save("Models/mask_detector.model", save_format="h5")

# Plotting the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")