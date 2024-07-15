import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from PIL import Image

model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

st.markdown(
    """
    <style>
    .custom-text-input {
        padding: 15px;
        font-size: 20px;
        color: #4a4a4a;
        background-color: #caf0f8;
        border: 5px solid #e5e5e5;
        border-radius: 5px;
        width: 100%;
        box-sizing: border-box;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a custom text input using HTML
review = st.text_input(label='Enter movie review:', placeholder='Enter text here...', key='custom_input')

# Inject the custom style class into the text input
st.markdown(
    """
    <script>
    const inputBox = window.parent.document.querySelector('input[key="custom_input"]');
    inputBox.classList.add('custom-text-input');
    </script>
    """,
    unsafe_allow_html=True
)

# Display the user input
st.write("""
        <style>
        .input-text {
            color: #ffffff;
            font-size: 15px;
            font-weight: bold;
            line-height: .5px
        }
        </style>
        <p class="input-text">User input:</p>
        """ ,review, unsafe_allow_html=True)

st.write()


# review = st.text_input('Enter Moview Review')

if st.button('Predict'):
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    if result[0] == 0:
        negative_img = Image.open("negative.png")
        st.write("""
        <style>
        .custom-text {
            color: #ffe5ec;
            font-size: 30px;
            font-weight: bold;
        }
        </style>
        <p class="custom-text">Negative review</p>
        """, negative_img , unsafe_allow_html=True)
    else:
        positive_img = Image.open("positive.png")
        st.write("""
        <style>
        .custom-text {
            color: #ffe5ec;
            font-size: 30px;
            font-weight: bold;
        }
        </style>
        <p class="custom-text">Positive review</p>
        """, positive_img , unsafe_allow_html=True)
