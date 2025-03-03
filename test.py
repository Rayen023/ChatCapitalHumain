import pandas as pd
import streamlit as st

data = {
    "School": [
        "Aux quatre vents",
        "Centre La Fontaine",
        "Louis-Mailloux",
        "Marie-Esther",
        "Roland-Pépin",
        "Secondaire Népisiguit",
        "W.-A.-Losier",
    ],
    "Female": [7, 3, 13, 4, 8, 22, 23],
    "Male": [1, 0, 3, 2, 7, 8, 4],
}

df = pd.DataFrame(data)
df = df.set_index("School")

st.dataframe(df)

st.bar_chart(df)
