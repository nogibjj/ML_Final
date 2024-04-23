import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L

# %%
data = pd.read_csv("FINAL_DATA_COMPLETE.csv")

# %%
data

# %%
data["id_image"] = data["id_image"].str.strip()

# %%
eval(data["target"][0])

# %%
index_to_delete = data[data["target"] == "[nan, nan, nan, nan]"].index

# Delete rows with the specified index
data.drop(index=index_to_delete, inplace=True)

# %%
data["target"] = data["target"].apply(lambda x: eval(x))

# %%
data["target"]

# %% [markdown]
# ## Train Test Split

# %% [markdown]
# ### Make sure for each category the train and test split is done equally

# %%
"""
X = data["id_image"]  # Features
y = data["target"]  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create DataFrames for training and testing data
train_df = pd.concat(
    [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
)
"""

# %%
food_subset = data["key_value"].unique().tolist()

# %%
# Initialize lists to hold train and test data
train_data = []
test_data = []

# Iterate over each unique category
for category in food_subset:
    # Filter the DataFrame for the current category
    category_data = data[data["key_value"] == category]

    # Split the category data into train and test sets while ensuring equal representation
    train_cat, test_cat = train_test_split(
        category_data, test_size=0.1, random_state=42
    )

    # Add the train and test data to the respective lists
    train_data.append(train_cat)
    test_data.append(test_cat)

# Concatenate the train and test data for each category
train_df = pd.concat(train_data)
test_df = pd.concat(test_data)

# %%
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# %%
train_df = train_df[["id_image", "target"]]
test_df = test_df[["id_image", "target"]]

# %%
# train_df.columns = ['Target_' + str(i) for i in range(1, len(train_df.columns)+1)]
df_split = pd.DataFrame(
    train_df["target"].tolist(), columns=[f"{'target'}_{i+1}" for i in range(4)]
)

# Combine the split columns with the original DataFrame
train_df = pd.concat([train_df.drop(columns=["target"]), df_split], axis=1)

# %%
train_df["target_1"]

# %%
# train_df.columns = ['Target_' + str(i) for i in range(1, len(train_df.columns)+1)]
df_split = pd.DataFrame(
    test_df["target"].tolist(), columns=[f"{'target'}_{i+1}" for i in range(4)]
)

# Combine the split columns with the original DataFrame
test_df = pd.concat([test_df.drop(columns=["target"]), df_split], axis=1)

# %%
train_df

# %%
K.clear_session()

# %%
bestmodel_path = "bestmodel_Calories.keras"
history_path = "history_Calories.log"

# %%
training_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
)

# %%
testing_data = ImageDataGenerator(preprocessing_function=preprocess_input)

# %%
training_data_generator = training_data.flow_from_dataframe(
    dataframe=train_df,
    directory="/Users/udyansachdev/Downloads/Recipes5k/images",
    x_col="id_image",
    y_col=["target_1", "target_2", "target_3", "target_4"],
    target_size=(300, 300),
    batch_size=8,
    class_mode="raw",
)

# %%
validation_data_generator = testing_data.flow_from_dataframe(
    dataframe=test_df,
    directory="/Users/udyansachdev/Downloads/Recipes5k/images",
    x_col="id_image",
    y_col=["target_1", "target_2", "target_3", "target_4"],
    target_size=(300, 300),
    batch_size=8,
    class_mode="raw",
)

# %%
inception = InceptionV3(weights="imagenet", include_top=False)
fc = inception.output
fc = GlobalAveragePooling2D()(fc)
fc = Dense(128, activation="relu")(fc)
fc = Dropout(0.2)(fc)

predictions = Dense(4, kernel_regularizer=regularizers.l2(0.005), activation="softmax")(
    fc
)

model = Model(inputs=inception.input, outputs=predictions)
model.compile(
    optimizer=SGD(learning_rate=0.0001, momentum=0.9),
    # loss="categorical_crossentropy",
    loss="mean_squared_error",
    metrics=["mse"],
)

# %%
checkpoint = ModelCheckpoint(filepath=bestmodel_path, verbose=1, save_best_only=True)
csv_logger = CSVLogger(history_path)

# %%
history = model.fit(
    training_data_generator,
    steps_per_epoch=training_data_generator.n // 8,
    validation_data=validation_data_generator,
    validation_steps=validation_data_generator.n // 8,
    epochs=20,
    verbose=1,
    callbacks=[csv_logger, checkpoint],
)

# %%
model.save("trainedmodel_Calories.keras")

# %%
import matplotlib.pyplot as plt


def plot_training_history(history, title, ylim=None):
    plt.title(title)
    plt.plot(history.history["mse"], label="Training MSE")
    plt.plot(history.history["val_mse"], label="Validation MSE")
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    if ylim:
        plt.ylim(ylim)
    plt.legend(loc="best")
    plt.show()


def plot_loss_history(history, title, ylim=None):
    plt.title(title)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    if ylim:
        plt.ylim(ylim)
    plt.legend(loc="best")
    plt.show()


# Example usage:
# Set custom y-axis range for MSE and Loss plots
ylim_mse = (0, 0.08)  # Example range for MSE plot
ylim_loss = (0, 0.1)  # Example range for Loss plot

plot_training_history(history, "Training and Validation Accuracy", ylim=ylim_mse)
plot_loss_history(history, "Training and Validation Loss", ylim=ylim_loss)

# %%
model_best = load_model("trainedmodel_Calories.keras", compile=False)

# %% [markdown]
# ## Prediction

# %%
img = "/Users/udyansachdev/Downloads/ML_Final/test_images/2.jpeg"
# apple_pie/36_vermont_apple_slab_pie_hostedLargeUrl.jpg
# apple_pie/31_skinny_apple_pie_hostedLargeUrl.jpg
# apple_pie/2_perfect_apple_pie_hostedLargeUrl.jpg

# %%
img = image.load_img(img, target_size=(300, 300))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# %%
pred = model_best.predict(img)[0]

# %%
pred
