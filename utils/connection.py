# utils/connection.py
import asyncio
import threading

import motor.motor_asyncio

from utils.utils import get_env_variable

# Create a global background event loop
_background_loop = asyncio.new_event_loop()


def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


threading.Thread(
    target=start_background_loop, args=(_background_loop,), daemon=True
).start()

# Create the Motor client using the background loop
client = motor.motor_asyncio.AsyncIOMotorClient(
    get_env_variable("MONGO_URI"), io_loop=_background_loop
)
db = client["capitalhumain_db"]
collection = db["capitalhumain_convs"]
