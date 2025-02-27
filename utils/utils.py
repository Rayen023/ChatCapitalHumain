import logging
import os

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_llm(MODEL_CONFIG):
    model = MODEL_CONFIG.get("model_name")
    if model is None:
        raise ValueError("No model specified in MODEL_CONFIG")

    config = MODEL_CONFIG.copy()
    config.pop("model_name", None)

    if model.startswith("google/"):
        model = model[len("google/") :]
        return ChatGoogleGenerativeAI(model=model, **config)
    elif model.startswith("anthropic/"):
        model = model[len("anthropic/") :]
        return ChatAnthropic(model="claude-3-7-sonnet-20250219", **config)
    else:
        return ChatOpenAI(
            model_name=model,
            openai_api_key=st.secrets["OPENROUTER_API_KEY"],
            openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
            **config
        )


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
