# app/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    mongodb_uri:   str = Field(..., env="MONGODB_URI")
    database_name: str = Field(..., env="DATABASE_NAME")

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"

settings = Settings()
