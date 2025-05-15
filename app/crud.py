import motor.motor_asyncio
from .config import settings
from bson import ObjectId

# set up Motor client & get handles
client   = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_uri)
db       = client[settings.database_name]
fs       = motor.motor_asyncio.AsyncIOMotorGridFSBucket(db)

async def save_file(file_bytes: bytes, filename: str) -> ObjectId:
    """
    Save raw bytes into GridFS, return the file ID.
    """
    file_id = await fs.upload_from_stream(filename, file_bytes)
    return file_id

async def save_metadata(doc: dict) -> ObjectId:
    """
    Insert metadata (including reference to file_id).
    """
    result = await db.metadata.insert_one(doc)
    return result.inserted_id
