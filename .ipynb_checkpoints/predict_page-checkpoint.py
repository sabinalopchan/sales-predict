from unicodedata import category
import streamlit as st
import pickle
import numpy as np
import sklearn


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
        return data


data = load_model()

regressor = data["model"]
le_city = data["le_city"]
le_state = data["le_state"]
le_category = data["le_category"]
le_sub_category = data["le_sub_category"]


# Function to handle unseen labels
def transform_with_unseen_labels(encoder, values):
    unseen_value = len(encoder.classes_)
    classes = list(encoder.classes_) + ['<unseen>']
    encoder.classes_ = np.array(classes)
    return np.array([encoder.transform([item if item in encoder.classes_ else '<unseen>'])[0] for item in values])


def show_predict_page():
    st.title("Retail Sales Prediction")

    st.write("""### We need some information to predict the sales""")

    cities = (
        "Springfield",
        "Columbus",
        "Auburn",
        "Roswell",
        "Marion",
        "Sudbury",
        "Tewksbury",
        "Fairhaven",
        "Killingly Center",
        "Needham"
    )

    states = (
        "California",
        "Texas",
        "Illinois",
        "Florida",
        "Ohio",
        "New York",
        "Michigan",
        "Indiana",
        "Washington",
        "Minnesota",
        "Pennsylvania",
        "North Carolina",
        "Virginia",
        "Georgia",
        "Maryland",
        "New Jersey",
        "Colorado",
        "Wisconsin",
        "Oregon",
        "Tennessee",
        "MO",
        "Iowa",
        "MA",
        "Utah",
        "Arizona",
        "Kansas",
        "Maine",
        "Alabama",
        "Arkansas",
        "Idaho",
        "South Carolina",
        "Oklahoma"
    )

    categories = (
        "Office Supplies",
        "Furniture",
        "Technology"
    )

    sub_categories = (
        'Storage & Organization',
        'Office Furnishings',
        'stationary',
        'Office Machines',
        'Computer Peripherals',
        'Appliances',
        'accessories'
    )

    city = st.selectbox("City", cities)
    state = st.selectbox("State", states)
    category = st.selectbox("Product Category", categories)
    sub_category = st.selectbox("Product Sub-Category", sub_categories)

    quantity = st.slider("Product Quantity", 0, 50, 3)

    ok = st.button("Predict Sales")

    if ok:
        # X_new = np.array([[quantity, category,
        #                    state]])
        X_new = np.array([[city, quantity, category,
                         sub_category, state]])
        X_new[:, 0] = transform_with_unseen_labels(le_city, X_new[:, 0])
        X_new[:, 2] = transform_with_unseen_labels(le_category, X_new[:, 2])
        X_new[:, 3] = transform_with_unseen_labels(
            le_sub_category, X_new[:, 3])
        X_new[:, 4] = transform_with_unseen_labels(le_state, X_new[:, 4])
        X_new = X_new.astype(float)

        sales = regressor.predict(X_new)
        st.subheader(f"The estimated sales is ${sales[0]:.2f}")
