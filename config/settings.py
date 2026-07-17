import os 
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"