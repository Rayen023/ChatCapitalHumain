import asyncio
import threading
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from utils.connection import collection


def message_to_dict(msg):
    """Convert a message object into a dict with role and content."""
    if isinstance(msg, AIMessage):
        return {"role": "assistant", "content": msg.content}
    elif isinstance(msg, HumanMessage):
        return {"role": "human", "content": msg.content}
    return {}


# Create and start a background event loop
_background_loop = asyncio.new_event_loop()


def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


threading.Thread(
    target=start_background_loop, args=(_background_loop,), daemon=True
).start()


async def save_chat_logs_async(email, messages, thread_id):
    messages_to_save = [message_to_dict(msg) for msg in messages]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = await collection.update_one(
        {"_id": thread_id},
        {
            "$set": {
                "timestamp": timestamp,
                "messages": messages_to_save,
                "email": email,
            }
        },
        upsert=True,
    )
    # Optionally log the result to verify the save:
    print("Save result:", result.raw_result)


def save_chat_logs():
    # Capture the required state in the main thread
    email = st.experimental_user.get("email")
    if not email:
        return  # Ensure user is logged in

    messages = st.session_state.get("messages", [])
    thread_id = st.session_state.get("thread_id")

    # Schedule the coroutine on the background event loop
    future = asyncio.run_coroutine_threadsafe(
        save_chat_logs_async(email, messages, thread_id), _background_loop
    )
    # Optionally, wait for the result or log errors
    try:
        result = future.result(
            timeout=5
        )  # Wait up to 5 seconds for the save to complete
    except Exception as e:
        # Log the exception if saving fails
        print("Async save failed:", e)
