# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns

from json import load
import sys
print(sys.executable)


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_sub_category(x):
    if 'Storage & Organization' in x:
        return 'Storage & Organization'
    if 'Chairs & Chairmats' in x or 'Tables' in x or 'Office Furnishings' in x:
        return 'Office Furnishings'
    if 'Paper' in x or 'Binders and Binder Accessories' in x or 'Pens & Art Supplies' in x or 'Bookcases' in x or 'Copiers and Fax' in x or 'Labels' in x or 'Envelopes' in x or 'Scissors, Rulers and Trimmers' in x:
        return 'stationary'
    if 'Office Machines' in x or 'Telephones and Communication' in x:
        return 'Office Machines'
    if 'Computer Peripherals' in x:
        return 'Computer Peripherals'
    if 'Appliances' in x:
        return 'Appliances'
    if 'Rubber Bands' in x:
        return 'accessories'


st.cache_data


def load_data():
    df = pd.read_csv("walmart_sales.csv")
    df = df[["City", "Order Quantity", "Product Category",
             "Product Sub-Category", "Sales", "State"]]
    df = df.dropna()
    df.isnull().sum()
    df['State'].value_counts()
    df['City'].value_counts()

    country_map = shorten_categories(df.State.value_counts(), 100)
    df['State'] = df['State'].map(country_map)
    df.State.value_counts()

    return df


df = load_data()


def show_explore_page():
    st.title("Explore Sales")

    st.write(
        """
     ### Stack Overflow Sales Survey
     """
    )

    data = df["State"].value_counts()
    # fig1, ax1 = plt.subplots()
    # ax1.pie(data, labels=data.index, autopct="%1.1f%%",
    #         shadow=True, startangle=90)
    # # Equal aspect ratio ensures that pie is drawn as a circle.
    # ax1.axis("equal")

    # st.write("""#### Number of Data from different countries""")

    # st.pyplot(fig1)

    st.write(
        """
    #### Mean Sales Based On State
    """
    )
    data = df.groupby(["State"])["Sales"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Sales Based On Category
    """
    )

    data = df.groupby(["Product Category"])[
        "Sales"].mean().sort_values(ascending=True)
    st.line_chart(data)

    st.write("""
    #### Mean Sales Based On Category
    """)

    mean_sales = df.groupby("Product Sub-Category")[
        "Sales"].mean().sort_values(ascending=True)
    st.area_chart(mean_sales)

    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=['number'])

    corr_matrix = numeric_df.corr()

    st.write("""
    #### Correlation Heatmap
    """)

    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
