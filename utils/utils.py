import logging
import os

import streamlit as st

logging.basicConfig(
    filename="logs.log",
    encoding="UTF-8",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Helper to get environment variables
def get_env_variable(var_name):
    try:
        if var_name in os.environ:
            return os.environ[var_name]
        if var_name in st.secrets:
            return st.secrets[var_name]
    except Exception as e:
        logger.error(
            "An error occurred retrieving the environment variable %s: %s",
            var_name,
            str(e),
        )
    return None
