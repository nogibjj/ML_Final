import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import plotly.express as px
import pandas as pd
import random

# Set page layout to wide
st.set_page_config(layout="wide")


# Define function to load and preprocess image
def load_and_preprocess_image(image_file, target_size):
    img = Image.open(image_file)
    img = img.resize(target_size).convert("RGB")
    img = np.array(img)
    return img


# Define function to preprocess calories image
def preprocess_calories(image_file, target_size):
    img = image.load_img(image_file, target_size=target_size, color_mode="rgb")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Define function to preprocess ingredients image
def preprocess_ingredients(image_file, target_size):
    img = image.load_img(image_file, target_size=target_size, color_mode="rgb")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# Define function to predict dish label
def predict_class(model, image):
    food_subset = [
        "tiramisu",
        "tuna_tartare",
        "beet_salad",
        "fish_and_chips",
        "pancakes",
        "caesar_salad",
        "garlic_bread",
        "carrot_cake",
        "chocolate_mousse",
        "hot_dog",
        "steak",
    ]
    food_subset.sort()

    img_array = np.expand_dims(image, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = food_subset[predicted_class_index]
    return predicted_class


# Define function to create pie chart for calories breakdown
def create_pie_chart(predictions):
    categories = ["Fat", "Carb", "Protein"]
    fig = px.pie(values=predictions, names=categories)
    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="center", x=0)
    )
    return fig


# Define function to predict top ingredients
def predict_top_ingredients(model, image, target_val):
    pred = model.predict(image)
    top_indices = np.argsort(pred[0])[::-1][:10]  # Get indices of top 10 ingredients
    top_ingredients = [
        target_val.iloc[i][0] for i in top_indices
    ]  # Assuming target_val is a DataFrame
    return top_ingredients


# Main function to create Streamlit app
def main():
    # Set title and description
    st.title("Dish Detective")

    # Upload image
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        img = Image.open(uploaded_file)
        resized_img = img.resize((250, 200))
        col2.write("#### Uploaded Image:")
        col2.image(resized_img, caption="Uploaded Image", use_column_width=True)

        # Load pre-trained models
        model_label = load_model("bestmodel.keras", compile=False)
        model_calories = load_model("trainedmodel_Calories.keras", compile=False)
        model_ingredients = load_model("bestmodel_ingredient.keras", compile=False)

        # Preprocess the image for label prediction
        processed_image = load_and_preprocess_image(uploaded_file, (299, 299))

        # Preprocess the image for calories prediction
        processed_calories = preprocess_calories(uploaded_file, (300, 300))

        # Predict dish label
        predicted_class_name = predict_class(model_label, processed_image)

        # Predict calories breakdown
        pred_calories = model_calories.predict(processed_calories)[0][:3]

        # Predict top ingredients
        target_val = pd.read_csv("ingredients.csv")  # Load ingredient data
        processed_calories_breakdown = preprocess_ingredients(uploaded_file, (300, 300))
        top_ingredients = predict_top_ingredients(
            model_ingredients, processed_calories_breakdown, target_val
        )

        # Display the predicted label
        col4.write("#### Predicted Dish:")
        col4.write(predicted_class_name)

        # Display the pie chart for predicted calories
        col4.write("#### Calories Breakdown:")
        fig = create_pie_chart(pred_calories)
        col4.plotly_chart(fig)

        # Display top predicted ingredients in a table
        col6.write("#### Predicted Ingredients:")

        df = pd.DataFrame({"Ingredient": top_ingredients})
        ingredients = pd.read_csv(
            "/Users/udyansachdev/Downloads/ML_Final/FINAL_DATA_COMPLETE.csv"
        )
        ingredients = ingredients[ingredients["key_value"] == predicted_class_name]
        column_values = ingredients["ingredients"]
        values_list = [
            value.strip()
            for sublist in column_values.str.split(",")
            for value in sublist
        ]
        unique_values = list(set(values_list))
        y = df["Ingredient"].to_list()

        intersection = set(y).intersection(unique_values)

        # Filter list1 to keep only elements present in list2
        list1 = [x for x in y if x in intersection]

        df2 = pd.DataFrame({"Ingredient": list1})

        col6.table(df2)


if __name__ == "__main__":
    main()
