# # <span style="color:red">Important</span>
#
# Go to section 2 for the function that retrieves the distribution.
#
# Go to section 3 if you just want to upload the data ready to go.
#
# # 1. ETL
#
#
#
# 5k as the only data source. You can download the data [here](http://www.ub.edu/cvub/recipes5k/).

# %% [markdown]
# ## 1.1 Uploading data train_images.txt, recepies and indexes

# %%
import pandas as pd

# Importing training images ID
file_path = "5k/Recipes5k/annotations/train_images.txt"
trained_images = pd.read_csv(file_path, sep="\t", names=["id_image"])
print(f"N elements in training: {trained_images.shape[0]}")

# Importing labels
file_path = "5k/Recipes5k/annotations/train_labels.txt"
trained_labels = pd.read_csv(file_path, sep="\t", names=["index_recepies"])

print(f"N elements in labels: {trained_labels.shape[0]}")

# importing recepies
file_path = "5k/Recipes5k/annotations/ingredients_simplified_Recipes5k.txt"
ingredients_index = pd.read_csv(file_path, sep="\t", names=["index_ingredients"])

print(f"N ingredients in indices in index: {ingredients_index.shape[0]}")

i = 34

print(trained_images.iloc[i])
print(trained_labels.iloc[i])

ingredients_index.iloc[trained_labels.iloc[i]]["index_ingredients"]

# %%
# Splitting the names in label, name, key and separating

trained_images["label"] = trained_labels["index_recepies"]
trained_images["image_name"] = [i.split("/")[1] for i in trained_images["id_image"]]
trained_images["key_value"] = [i.split("/")[0] for i in trained_images["id_image"]]
trained_images["ingredients"] = [
    ingredients_index.iloc[i]["index_ingredients"] for i in trained_images["label"]
]

# %% [markdown]
# # Open AI prompt engineering for recepies

# %%
import os
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain
from langchain_openai import OpenAI


template = """I'll pass you a key representation name of a dish\
    and a list of ingredients such as:\
    key representation: apple_pie\
    list of ingredients:sugar,nutmeg,milk,butter,flour,pastry,cinnamon,apples,lemon\
    and you have to return a json as follow:\
    {{"{key}": {{"sugar": 150,"nutmeg": 0.50,"milk": 240,"butter": 113,"flour": 180,\
    "pastry": 1, "cinnamon": 2.60,"apples": 680,"lemon": 15}}}}\
    where each ingredient is listed with its corresponding quantity in grams.\
    Now do it but with:\
    key representation: {key}\
    list of ingredients: {ingredients}
"""

# %% [markdown]
# ## Using OpenAI to bring the recepies in grams
#
# Checking how much it will cost knowing that is 6 dollar per 1M tokens

# %%
# from langchain.callbacks import get_openai_callback

# for _, row in trained_images.iterrows():

#     with get_openai_callback() as cb:
#         # Your code here, for example:
#         prompt = PromptTemplate(input_variables=["key", "ingredients"], template=template)
#         llm_gpt3_5_chain = LLMChain(prompt=prompt, llm=OpenAI())
#         response = llm_gpt3_5_chain.run(
#             key=row["key_value"],
#             ingredients=row["ingredients"],
#             temperature=0,
#         )


#     total_tokens = cb.total_tokens
#     print(f"Total tokens used: {total_tokens}")
#     break

# %% [markdown]
# ### Getting the right format and grams using OpenAI

# %%
# import json

# recepies_grams = []
# counter = 1

# for _, row in trained_images.iterrows():

#     prompt = PromptTemplate(input_variables=["key", "ingredients"], template=template)
#     llm_gpt3_5_chain = LLMChain(prompt=prompt, llm=OpenAI())

#     input_params = {
#         "key": row["key_value"],
#         "ingredients": row["ingredients"],
#         "temperature": 0,
#     }
#     response = llm_gpt3_5_chain.invoke(input_params)
#     recepies_grams.append(response["text"])

#     counter += 1
#     if counter % 50 == 0:
#         print(f"Number: {counter}")

# trained_images['json_format']=recepies_grams
# trained_images.to_csv('FINAL_DATA.csv',index=False)

# %% [markdown]
# ## 1.2 Read the data with the OpenAI retrievals

# %%
data = pd.read_csv("FINAL_DATA.csv")

# %%
data["json_format"] = [
    i.replace("\n\n", ",").replace("\n", "") for i in data["json_format"]
]

# %%
import json
import ast

as_dic = []
counter = 0
indexes = []

for i, row in data.iterrows():
    string = row["json_format"]
    try:
        data_list = ast.literal_eval(string)

        if len(data_list) == 1:
            result = json.loads(string)
            as_dic.append(result)

        elif len(data_list) >= 2:
            result = dict([list(ast.literal_eval(string).items())[1]])
            as_dic.append(result)

    except:
        indexes.append(i)

# %%
### How many data had trouble with the API:

len(indexes)

# %%
# How is the distributiono data with problems:

data[["key_value", "label"]].iloc[indexes].groupby(
    "key_value", as_index=False
).count().sort_values("label", ascending=False)

# %%
data = data.drop(indexes)

# %%
data["json_format_clean"] = as_dic


# %%
def sum_nested_dict_values(nested_dict):
    total_sum = 0
    for inner_dict in nested_dict.values():
        total_sum += sum(inner_dict.values())
    return total_sum


indexes = []
total_g = []

for i, row in data.iterrows():
    dic_to_transform = row["json_format_clean"]
    try:
        total_g.append(sum_nested_dict_values(dic_to_transform))
    except:
        indexes.append(i)

# %%
data = data.drop(indexes)
data["total_g"] = total_g

# %%
for i, row in data[["json_format_clean", "total_g"]].sample(2).iterrows():
    print(row["json_format_clean"])
    print(row["total_g"])

# %% [markdown]
# # 2. Getting the proportions of fat, carbs and protein
#
# In this section we are creating the nutritient proportions using 5k.
#
# ## VectorDatabase
#
# As some ingredients are not in out nutrition dataframe we assing the closest based on similarity embeddings. If the query is empty, zero distribution of nutrients are assigned. For example, this happen with *cold water*.

# %%
import chromadb

chroma_client = chromadb.Client()

# Create or load a collection
collection_name = "mycollection"

# Get all collections
collections = chroma_client.list_collections()

# Extract names from collection objects
collection_names = [collection.name for collection in collections]

# Check if the collection exists
if collection_name in collection_names:
    # Load the existing collection
    collection = chroma_client.get_collection(collection_name)
    print("Loaded existing collection:", collection_name)
else:
    # Create a new collection and add data to it
    collection = chroma_client.create_collection(name=collection_name)
    print("Created new collection:", collection_name)

# %%
from sentence_transformers import SentenceTransformer

# Read the ingredients:
file_path = "nutrition5k_dataset_metadata_ingredients_metadata.csv"
nutrition = pd.read_csv(
    file_path,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_list = model.encode(list(nutrition["ingr"]))

# %%
# How json cleaned look like
data["json_format_clean"][0]

# %%
# How nutrition dataframe look like
nutrition.sample(4)

# %%
# Number of ingredients
nutrition.shape[0]

# %% [markdown]
# ## Adding the nutrients into the vector database

# %%
collection.add(
    embeddings=embedding_list,
    documents=list(nutrition["ingr"]),
    ids=["ID" + str(i) for i in range(len(embedding_list))],
)

# %% [markdown]
# For example, "back ribs" is not in our ingredients, however only ribs it is, so it is assigned instead from the query to the vectordatabase.

# %%
"back ribs" in set(nutrition["ingr"])

# %%
label = list(data["json_format_clean"][0].keys())[0]
ingredients = list(data["json_format_clean"][0][label].keys())

# %%
# Sample of nutrition loaded in the first section
nutrition.head(3)

# %% [markdown]
# ### This function return the distribution after the repepies are cleaned.

# %%
# Calories, fat, carb and protein


def get_calories_fat_carb_protein(dictionary_of_ingredients, nutrition, total_grams):
    """This function return the distribution of nutritional information in a 4D vector with fat,carbs, and protein.
    In addition, return the ingredients where the nutrition retrieval was unsuccesful, and the labels additioning the total calories per portion.


    Args:
        dictionary_of_ingredients (dict): The dictionary with ingredient of a single recepy
        nutrition (dataframe): The data frame with the nutrition distribution
        total_grams (_type_): The addition of grams by portion

    Returns:

        json: with the labels and names of nutrimental
        ingredients_not_found (list): list of ingredients not found in the VDB
        vector: a 4D list with the nutrimental distribution
    """

    ingredients_not_found = []

    label = list(dictionary_of_ingredients.keys())[0]

    ingredients = list(dictionary_of_ingredients[label].keys())

    # Initializing variables
    cal = 0
    fat = 0
    carb = 0
    protein = 0

    for ingredient in ingredients:

        ingr = ingredient

        if ingredient in set(nutrition["ingr"]):

            cal += (
                nutrition[nutrition["ingr"] == ingr]["cal/g"].iloc[0]
                * dictionary_of_ingredients[label][ingredient]
            )
            fat += (
                nutrition[nutrition["ingr"] == ingr]["fat(g)"].iloc[0]
                * dictionary_of_ingredients[label][ingredient]
            )
            carb += (
                nutrition[nutrition["ingr"] == ingr]["carb(g)"].iloc[0]
                * dictionary_of_ingredients[label][ingredient]
            )
            protein += (
                nutrition[nutrition["ingr"] == ingr]["protein(g)"].iloc[0]
                * dictionary_of_ingredients[label][ingredient]
            )
        else:

            # print(f"ingredient out of nutrition: {ingredient}")

            similar_documents = collection.query(query_texts=[ingredient], n_results=1)

            if len(similar_documents["documents"][0]) == 0:
                ingredients_not_found.append(ingr)

            else:
                ingr = similar_documents["documents"][0][0]

                cal += (
                    nutrition[nutrition["ingr"] == ingr]["cal/g"].iloc[0]
                    * dictionary_of_ingredients[label][ingredient]
                )
                fat += (
                    nutrition[nutrition["ingr"] == ingr]["fat(g)"].iloc[0]
                    * dictionary_of_ingredients[label][ingredient]
                )
                carb += (
                    nutrition[nutrition["ingr"] == ingr]["carb(g)"].iloc[0]
                    * dictionary_of_ingredients[label][ingredient]
                )
                protein += (
                    nutrition[nutrition["ingr"] == ingr]["protein(g)"].iloc[0]
                    * dictionary_of_ingredients[label][ingredient]
                )

    vector = [
        fat / total_grams,
        carb / total_grams,
        protein / total_grams,
        1 - (fat + carb + protein) / total_grams,
    ]

    return (
        {
            label: {
                "cal": cal,
                "fat": fat / total_grams,
                "carb": carb / total_grams,
                "protein": protein / total_grams,
                "other": 1 - (fat + carb + protein) / total_grams,
            }
        },
        ingredients_not_found,
        vector,
    )


# %%
list_distribution = []
list_jsons = []
list_not_found = []


for i, row in data.iterrows():

    j, n, v = get_calories_fat_carb_protein(
        dictionary_of_ingredients=row["json_format_clean"],
        nutrition=nutrition,
        total_grams=row["total_g"],
    )
    list_distribution.append(v)
    list_jsons.append(j)
    list_not_found.append(n)

# %%
data["target"] = list_distribution
data["json_nutrition"] = list_jsons
data["ing_not_found"] = list_not_found

# %% [markdown]
# # 3. Reading the DATA for Neural Network with Keras

# %%
# data.to_csv('FINAL_DATA_COMPLETE.csv',index=False)

data = pd.read_csv("FINAL_DATA_COMPLETE.csv")

# %%
list_of_pics = os.listdir("5k/Recipes5k/images/SAMPLE/final_images")

# %%
list_of_pics = list(set(list_of_pics).intersection(set(data["image_name"])))
len(list_of_pics)

# %%
data["target"][
    data["image_name"] == "34_honey_sriracha_chicken_wings_hostedLargeUrl.jpg"
]

# %%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import BatchNormalization


image_directory = "5k/Recipes5k/images/SAMPLE/final_images"

# %%
SIZE = 200
X_dataset = []
for i in list_of_pics:
    img = image.load_img(
        "5k/Recipes5k/images/SAMPLE/final_images/" + i, target_size=(SIZE, SIZE, 3)
    )
    img = image.img_to_array(img)
    img = img / 255.0
    X_dataset.append(img)

X = np.array(X_dataset)

y = [data["target"][data["image_name"] == i].tolist()[0] for i in list_of_pics]

y = np.array(y)

# %%
# Check for NaNs or infinity in the array
nan_indices = np.isnan(y).any(axis=1)
inf_indices = np.isinf(y).any(axis=1)

# Get the indices where NaNs or infinity occur
indices_with_nan = np.where(nan_indices)[0]
indices_with_inf = np.where(inf_indices)[0]

if len(indices_with_nan) > 0:
    print("NaNs found at indices:", indices_with_nan)

if len(indices_with_inf) > 0:
    print("Infinity values found at indices:", indices_with_inf)

# %%
indices_with_nan

# %%
# Dropping NANs

y = np.delete(y, indices_with_nan, axis=0)
X = np.delete(X, indices_with_nan, axis=0)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=20, test_size=0.3
)

# %%
len(y)

# %% [markdown]
# ## Keras Implementation

# %%
model = Sequential()

model.add(
    Conv2D(
        filters=16, kernel_size=(5, 5), activation="relu", input_shape=(SIZE, SIZE, 3)
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4, activation="softmax"))

# %%
model.summary()

# Categorical was better suited for probabilities.
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


history = model.fit(
    X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64
)


# plot the training and validation accuracy and loss at each epoch
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "y", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "y", label="Training acc")
plt.plot(epochs, val_acc, "r", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %%
images_path = "5k/Recipes5k/images/SAMPLE/final_images/"
os.listdir(images_path)[:2]

# %% [markdown]
# # Prediction

# %%
img = image.load_img(
    images_path + "11_peanut_butter_frozen_yogurt_hostedLargeUrl.jpg",
    target_size=(SIZE, SIZE, 3),
)

img = image.img_to_array(img)
img = img / 255.0
plt.imshow(img)
img = np.expand_dims(img, axis=0)
proba = model.predict(img)

# %%
# Real value

list(
    data["json_nutrition"][
        data["image_name"] == "11_peanut_butter_frozen_yogurt_hostedLargeUrl.jpg"
    ]
)

# %% [markdown]
# ## Distribution

# %%
proba
