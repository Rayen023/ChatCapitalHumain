import pandas as pd
import streamlit as st

# Data from the table
data = {
    "School Name": [
        "Aux quatre vents",
        "Aux quatre vents",
        "Centre La Fontaine",
        "Centre La Fontaine",
        "Louis-Mailloux",
        "Louis-Mailloux",
        "Marie-Esther",
        "Marie-Esther",
        "Roland-Pépin",
        "Roland-Pépin",
        "Secondaire Népisiguit",
        "Secondaire Népisiguit",
        "W.-A.-Losier",
        "W.-A.-Losier",
    ],
    "Gender": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
    "Number of Students": [7, 1, 3, 0, 13, 3, 4, 2, 8, 7, 22, 8, 23, 4],
}

df = pd.DataFrame(data)

st.title("Students Who Walked to School in 2018")

st.header("Data Table")
st.dataframe(df)

st.header("Number of Students by School and Gender")
st.bar_chart(df.set_index(["School Name", "Gender"])["Number of Students"])

st.header("Number of Students by Gender")
gender_counts = df.groupby("Gender")["Number of Students"].sum()
st.bar_chart(gender_counts)

st.header("Number of Students by School")
school_counts = df.groupby("School Name")["Number of Students"].sum()
st.bar_chart(school_counts)
