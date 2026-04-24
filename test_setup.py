import langchain
import pinecone
from langchain_google_genai import ChatGoogleGenerativeAI

import torch

print(f"LangChain version: {langchain.__version__}")
print(f"Torch version: {torch.__version__}")
print("All libraries loaded successfully!")