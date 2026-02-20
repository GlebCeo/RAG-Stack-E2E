from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    DATABASE_SYNC_URL: str
    GROQ_API_KEY: str
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
