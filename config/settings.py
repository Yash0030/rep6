from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    BACKEND_BEARER_TOKEN = os.getenv("BACKEND_BEARER_TOKEN")  # Fixed typo
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_HOST_NAME = os.getenv("PINECONE_HOST_NAME")
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")
    GROQ_API_KEY_1=os.getenv("GROQ_API_KEY_1")
    GROQ_API_KEY_2=os.getenv("GROQ_API_KEY_2")
    CEREBRAS_API_KEY=os.getenv("CEREBRAS_API_KEY")
    CEREBRAS_API_KEY_1=os.getenv("CEREBRAS_API_KEY_1")
    CEREBRAS_API_KEY_2=os.getenv("CEREBRAS_API_KEY_2")
    

    # DATABASE_URL = os.getenv("DATABASE_URL")
    
    def validate(self):
        """Validate that required environment variables are set"""
        required_vars = [
            "HUGGINGFACEHUB_ACCESS_TOKEN",
            "PINECONE_API_KEY",
            "PINECONE_INDEX_NAME"
        ]
        
        missing = [var for var in required_vars if not getattr(self, var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
settings = Settings()