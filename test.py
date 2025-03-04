import pandas as pd
import streamlit as st

data = {
    "École": [
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
    "Genre": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
    "Nombre d'étudiants": [7, 1, 3, 0, 13, 3, 4, 2, 8, 7, 22, 8, 23, 4],
}

df = pd.DataFrame(data)

st.dataframe(df)

st.subheader("Nombre d'étudiants allant à l'école à pied par école")
df_pivot = df.pivot(index="École", columns="Genre", values="Nombre d'étudiants").fillna(
    0
)
st.bar_chart(df_pivot)

st.subheader("Nombre d'étudiants allant à l'école à pied par genre")
gender_counts = df.groupby("Genre")["Nombre d'étudiants"].sum()
st.bar_chart(gender_counts)
