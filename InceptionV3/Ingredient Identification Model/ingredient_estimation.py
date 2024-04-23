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
import torch

# %% [markdown]
# ## Read and sort training data

# %%
path = "/Users/udyansachdev/Downloads/Recipes5k/annotations"

# %%
file_path = path + "/" + "train_images.txt"

# Read the text file line by line
with open(file_path, "r") as file:
    lines = file.readlines()

# Create DataFrame from lines
y_train_id = pd.DataFrame(lines, columns=["id"])

# %%
y_train_id

# %%
file_path = path + "/" + "train_labels.txt"

# Read the text file line by line
with open(file_path, "r") as file:
    lines = file.readlines()

# Create DataFrame from lines
y_train_label = pd.DataFrame(lines, columns=["label"])

# %%
y_train_label["label"] = y_train_label["label"].str.strip()
y_train_label

# %%
y_train = pd.concat([y_train_id, y_train_label], axis=1)

# %%
y_train["label"] = y_train["label"].astype(int)

# %%
y_train

# %% [markdown]
# ### Read ingredients

# %%
file_path = path + "/" + "ingredients_simplified_Recipes5k.txt"

# Read the text file line by line
with open(file_path, "r") as file:
    lines = file.readlines()

# Create DataFrame from lines
ingredients = pd.DataFrame(lines, columns=["ingredients"])

# %%
ingredients["ingredients"] = ingredients["ingredients"].str.strip()
ingredients

# %%
special_characters = [
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "+",
    "-",
    "/",
    "\\",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "[",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
    "!",
    '"',
    "'",
    "(",
    ")",
    ".",
]

for char in special_characters:
    ingredients["ingredients"] = ingredients["ingredients"].str.replace(char, "")

ingredients

# %%
ingredients["label"] = ingredients.index

# %% [markdown]
# ### Merge dataframe

# %%
y_train_final = y_train.merge(ingredients, how="left", on="label")

# %%
y_train_final["id"] = y_train_final["id"].str.strip()
y_train_final

# %%
"""
# Split ingredients into individual items
y_train_final["ingredients"] = y_train_final["ingredients"].str.split(",")

# One-hot encode the 'ingredients' column
encoded_data = y_train_final["ingredients"].str.join("|").str.get_dummies()

# Concatenate the original DataFrame with the encoded DataFrame
data_encoded = pd.concat([y_train_final, encoded_data], axis=1)
"""

# %%
# data_encoded

# %% [markdown]
# ## Loading and preprocessing the training data

# %%
# data_encoded["id"] = data_encoded["id"].str.strip()

# %%
"""
train_image = []
for i in data_encoded["id"]:
    img = image.load_img(
        ("/Users/udyansachdev/Downloads/Recipes5k/images" + "//" + i),
        target_size=(400, 400, 3),
    )
    img = image.img_to_array(img)
    img = img / 255
    train_image.append(img)
X = np.array(train_image)
"""

# %%
# X.shape

# %%
"""
plt.imshow(X[2000])
data_encoded["ingredients"][2000]
"""

# %%
# y = np.array(data_encoded.drop(["id", "label", "ingredients"], axis=1))

# %%
# y.shape

# %% [markdown]
# ## Read and sort validation data

# %%
file_path = path + "/" + "val_images.txt"

# Read the text file line by line
with open(file_path, "r") as file:
    lines = file.readlines()

# Create DataFrame from lines
y_val_id = pd.DataFrame(lines, columns=["id"])

# %%
y_val_id

# %%
file_path = path + "/" + "val_labels.txt"

# Read the text file line by line
with open(file_path, "r") as file:
    lines = file.readlines()

# Create DataFrame from lines
y_val_label = pd.DataFrame(lines, columns=["label"])

# %%
y_val_label["label"] = y_val_label["label"].str.strip()
y_val_label

# %%
y_val = pd.concat([y_val_id, y_val_label], axis=1)
y_val["label"] = y_val["label"].astype(int)
y_val["id"] = y_val["id"].str.strip()

y_val

# %%
y_val_final = y_val.merge(ingredients, how="left", on="label")

# %%
y_val_final

# %% [markdown]
# ##  Concat Dataframe of y_train and y_val

# %%
y = pd.concat([y_train_final, y_val_final], axis=0, ignore_index=True)

# %%
y

# %% [markdown]
# ## Loading and preprocessing the testing data

# %%
file_path = path + "/" + "test_images.txt"

# Read the text file line by line
with open(file_path, "r") as file:
    lines = file.readlines()

# Create DataFrame from lines
y_test_id = pd.DataFrame(lines, columns=["id"])

# %%
y_test_id["id"] = y_test_id["id"].str.strip()
y_test_id

# %%
file_path = path + "/" + "test_labels.txt"

# Read the text file line by line
with open(file_path, "r") as file:
    lines = file.readlines()

# Create DataFrame from lines
y_test_label = pd.DataFrame(lines, columns=["label"])

# %%
y_test_label["label"] = y_test_label["label"].str.strip()
y_test_label

# %%
y_test = pd.concat([y_test_id, y_test_label], axis=1)

# %%
y_test["label"] = y_test["label"].astype(int)

# %%
y_test

# %% [markdown]
# ### Merging with ingredients
#

# %%
y_test_final = y_test.merge(ingredients, how="left", on="label")

# %%
y_test_final

# %% [markdown]
# ## Merging Training and Testing Dataframe

# %% [markdown]
# ### Ensuring the split have uniform y one hot encoded

# %%
data = pd.concat([y, y_test_final], axis=0, ignore_index=True)

# %%
data

# %%
# Split ingredients into individual items
data["ingredients"] = data["ingredients"].str.split(",")

# One-hot encode the 'ingredients' column
encoded_data = data["ingredients"].str.join("|").str.get_dummies()

# Concatenate the original DataFrame with the encoded DataFrame
encoded_data = pd.concat([data, encoded_data], axis=1)

# %%
encoded_data

# %%
encoded_data[["category", "image_name"]] = encoded_data["id"].str.split(
    "/", expand=True, n=1
)

# %%
encoded_data

# %% [markdown]
# ## Train and test split

# %% [markdown]
# ### Make sure for each category the train and test split is done equally

# %%
food_subset = encoded_data["category"].unique().tolist()

# %%
# Initialize lists to hold train and test data
train_data = []
test_data = []

# Iterate over each unique category
for category in food_subset:
    # Filter the DataFrame for the current category
    category_data = encoded_data[encoded_data["category"] == category]

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
# train_df["category"].unique()

# %%
# test_df["category"].unique()

# %%
"""
X = encoded_data["id"]  # Features
y = encoded_data.drop(["id", "label", "ingredients"], axis=1)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"""

# %%
"""
train_df = pd.concat(
    [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
)
"""

# %%
train_df = train_df.drop(["label", "ingredients", "category", "image_name"], axis=1)
train_df = train_df.reset_index(drop=True)
train_df

# %%
test_df = test_df.drop(["label", "ingredients", "category", "image_name"], axis=1)
test_df = test_df.reset_index(drop=True)
test_df

# %%
target_val = test_df.columns.tolist()
target_val = target_val[1:]

# %% [markdown]
# ## Creating model

# %%
bestmodel_path = "bestmodel_ingredient_1.keras"
trainedmodel_path = "trainedmodel_ingredient_1.keras"
history_path = "history_ingredient_1.log"

# %%
# Define data augmentation with more techniques
training_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.3,
    zoom_range=0.3,
    rotation_range=30,  # Add rotation augmentation
    width_shift_range=0.1,  # Add width shift augmentation
    height_shift_range=0.1,  # Add height shift augmentation
    horizontal_flip=True,
)

# %%
testing_data = ImageDataGenerator(preprocessing_function=preprocess_input)

# %%
training_data_generator = training_data.flow_from_dataframe(
    dataframe=train_df,
    directory="/Users/udyansachdev/Downloads/Recipes5k/images",
    x_col="id",
    y_col=target_val,
    target_size=(300, 300),
    batch_size=8,
    class_mode="raw",
)

# %%
validation_data_generator = testing_data.flow_from_dataframe(
    dataframe=test_df,
    directory="/Users/udyansachdev/Downloads/Recipes5k/images",
    x_col="id",
    y_col=target_val,
    target_size=(300, 300),
    batch_size=8,
    class_mode="raw",
)

# %%
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)

# Define the model
inception = InceptionV3(weights="imagenet", include_top=False)
fc = inception.output
fc = GlobalAveragePooling2D()(fc)
fc = Dense(128, activation="relu")(fc)
fc = Dropout(0.2)(fc)

predictions = Dense(
    1013, kernel_regularizer=regularizers.l2(0.005), activation="softmax"
)(fc)


model = Model(inputs=inception.input, outputs=predictions)
model.compile(
    optimizer=SGD(learning_rate=0.00001, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy", "mse"],
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
model.save(trainedmodel_path)

# %% [markdown]
# ### Load Saved model

# %%
model_best = load_model("bestmodel_ingredient.keras", compile=False)

# %%
img = "/Users/udyansachdev/Downloads/ML_Final/test_images/3.jpeg"
img = image.load_img(img, target_size=(300, 300))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# %%
pred = model_best.predict(img)

# %%
pred

# %%
# Convert array to a PyTorch tensor
tensor = torch.tensor(pred)

# Get the indices of the top 10 maximum values
top10_indices = torch.topk(tensor, k=12).indices.tolist()

# %%
top10_indices[0]

# %%
for i in top10_indices[0]:
    print(target_val[i])
