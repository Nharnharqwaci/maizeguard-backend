from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv, find_dotenv
import os


load_dotenv(find_dotenv())

MONGODB_URL = os.getenv("MONGODB_URL")
print(
    "Mongo URL:",
    MONGODB_URL
)
DATABASE_NAME = os.getenv("DATABASE_NAME", "plantdb")

if not MONGODB_URL:
    raise ValueError("MONGODB_URL is not set — check your .env file")

client = AsyncIOMotorClient(
    MONGODB_URL,
    serverSelectionTimeoutMS=15000,
    connectTimeoutMS=15000
)
db = client[DATABASE_NAME]

users_collection = db["users"]
scans_collection = db["scans_history"]