import streamlit as st
import tensorflow as tf
import numpy as np

# Simple Nutrition Information Dictionary
nutrition_data = {
    'apple': {'Calories (kcal)': 52, 'Protein (g)': 0.3, 'Vitamins': 'Vitamin C, Vitamin A, Vitamin K'},
    'banana': {'Calories (kcal)': 105, 'Protein (g)': 1.3, 'Vitamins': 'Vitamin C, Vitamin B6'},
    'beetroot': {'Calories (kcal)': 43, 'Protein (g)': 1.6, 'Vitamins': 'Vitamin A, Vitamin C, Folate'},
    'bell pepper': {'Calories (kcal)': 31, 'Protein (g)': 1, 'Vitamins': 'Vitamin C, Vitamin A, Vitamin B6'},
    'cabbage': {'Calories (kcal)': 25, 'Protein (g)': 1.3, 'Vitamins': 'Vitamin C, Vitamin K, Folate'},
    'capsicum': {'Calories (kcal)': 20, 'Protein (g)': 1, 'Vitamins': 'Vitamin C, Vitamin A, Vitamin B6'},
    'carrot': {'Calories (kcal)': 41, 'Protein (g)': 0.9, 'Vitamins': 'Vitamin A, Vitamin K, Vitamin C'},
    'cauliflower': {'Calories (kcal)': 25, 'Protein (g)': 1.9, 'Vitamins': 'Vitamin C, Vitamin K, Folate'},
    'chilli pepper': {'Calories (kcal)': 40, 'Protein (g)': 1.9, 'Vitamins': 'Vitamin C, Vitamin A, Vitamin B6'},
    'corn': {'Calories (kcal)': 96, 'Protein (g)': 3.4, 'Vitamins': 'Vitamin C, Vitamin B6, Folate'},
    'cucumber': {'Calories (kcal)': 15, 'Protein (g)': 0.7, 'Vitamins': 'Vitamin K, Vitamin C'},
    'eggplant': {'Calories (kcal)': 25, 'Protein (g)': 1, 'Vitamins': 'Vitamin C, Vitamin K, Folate'},
    'garlic': {'Calories (kcal)': 149, 'Protein (g)': 6.4, 'Vitamins': 'Vitamin C, Vitamin B6, Manganese'},
    'ginger': {'Calories (kcal)': 80, 'Protein (g)': 1.8, 'Vitamins': 'Vitamin C, Vitamin B6, Magnesium'},
    'grapes': {'Calories (kcal)': 69, 'Protein (g)': 0.7, 'Vitamins': 'Vitamin C, Vitamin K'},
    'jalapeño': {'Calories (kcal)': 4, 'Protein (g)': 0.1, 'Vitamins': 'Vitamin C, Vitamin A'},
    'kiwi': {'Calories (kcal)': 61, 'Protein (g)': 1.1, 'Vitamins': 'Vitamin C, Vitamin K, Folate'},
    'lemon': {'Calories (kcal)': 29, 'Protein (g)': 1.1, 'Vitamins': 'Vitamin C, Vitamin B6'},
    'lettuce': {'Calories (kcal)': 15, 'Protein (g)': 1.4, 'Vitamins': 'Vitamin K, Vitamin A, Folate'},
    'mango': {'Calories (kcal)': 60, 'Protein (g)': 0.8, 'Vitamins': 'Vitamin C, Vitamin A, Vitamin B6'},
    'onion': {'Calories (kcal)': 40, 'Protein (g)': 1.1, 'Vitamins': 'Vitamin C, Vitamin B6, Folate'},
    'orange': {'Calories (kcal)': 62, 'Protein (g)': 1.2, 'Vitamins': 'Vitamin C, Folate, Vitamin A'},
    'paprika': {'Calories (kcal)': 289, 'Protein (g)': 14, 'Vitamins': 'Vitamin C, Vitamin A, Vitamin B6'},
    'pear': {'Calories (kcal)': 57, 'Protein (g)': 0.4, 'Vitamins': 'Vitamin C, Vitamin K'},
    'peas': {'Calories (kcal)': 81, 'Protein (g)': 5.4, 'Vitamins': 'Vitamin C, Vitamin K, Folate'},
    'pineapple': {'Calories (kcal)': 50, 'Protein (g)': 0.5, 'Vitamins': 'Vitamin C, Vitamin A, Vitamin B6'},
    'pomegranate': {'Calories (kcal)': 83, 'Protein (g)': 1.7, 'Vitamins': 'Vitamin C, Vitamin K, Folate'},
    'potato': {'Calories (kcal)': 77, 'Protein (g)': 2, 'Vitamins': 'Vitamin C, Vitamin B6, Potassium'},
    'raddish': {'Calories (kcal)': 16, 'Protein (g)': 0.7, 'Vitamins': 'Vitamin C, Folate, Vitamin B6'},
    'soy beans': {'Calories (kcal)': 173, 'Protein (g)': 16.6, 'Vitamins': 'Vitamin K, Folate, Vitamin C'},
    'spinach': {'Calories (kcal)': 23, 'Protein (g)': 2.9, 'Vitamins': 'Vitamin K, Vitamin A, Folate'},
    'sweetcorn': {'Calories (kcal)': 86, 'Protein (g)': 3.3, 'Vitamins': 'Vitamin C, Vitamin B6, Folate'},
    'sweetpotato': {'Calories (kcal)': 86, 'Protein (g)': 1.6, 'Vitamins': 'Vitamin A, Vitamin C, Vitamin B6'},
    'tomato': {'Calories (kcal)': 18, 'Protein (g)': 0.9, 'Vitamins': 'Vitamin C, Vitamin K'},
    'turnip': {'Calories (kcal)': 28, 'Protein (g)': 0.9, 'Vitamins': 'Vitamin C, Vitamin K, Folate'},
    'watermelon': {'Calories (kcal)': 30, 'Protein (g)': 0.6, 'Vitamins': 'Vitamin C, Vitamin A'}
}

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("🍏 Fruits Recognition and Nutrition Information 🍅")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("Fruits Recognition and Nutrition Information")
    st.markdown("""
        This system uses deep learning to identify fruits and vegetables from an image.
        Upload an image, and it will predict the item and provide nutritional information.
    """)
    st.image("home_img.jpg", use_column_width=True)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.markdown("""
        This dataset contains images of various fruits and vegetables:
        - **Fruits:** banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango
        - **Vegetables:** cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant
    """)
    st.subheader("Content")
    st.markdown("""
        The dataset contains three folders:
        - **train:** 100 images each
        - **test:** 10 images each
        - **validation:** 10 images each
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, use_column_width=True)
        if st.button("Predict"):
            st.snow()
            result_index = model_prediction(test_image)
            
            # Reading Labels
            with open("labels.txt") as f:
                content = f.readlines()
            label = [i.strip() for i in content]
            predicted_label = label[result_index]
            
            # Display prediction result
            st.success(f"Prediction: It is a **{predicted_label}**!")
            
            # Fetch and display nutrition info from the dictionary
            nutrition_info = nutrition_data.get(predicted_label)
            if nutrition_info:
                st.subheader("Nutritional Information")
                for key, value in nutrition_info.items():
                    st.markdown(f"**{key}:** {value}")
            else:
                st.warning("Sorry, nutrition information is not available for this item.")

# Add a footer
st.markdown("""
    <style>
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)
