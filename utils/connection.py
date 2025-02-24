# Initialize the async MongoDB client (cache this as needed)
import motor.motor_asyncio
import streamlit as st

from utils.utils import get_env_variable


@st.cache_resource
def get_mongo_collection():
    client = motor.motor_asyncio.AsyncIOMotorClient(get_env_variable("MONGO_URI"))
    db = client["capitalhumain_db"]
    return db["capitalhumain_convs"]


collection = get_mongo_collection()
