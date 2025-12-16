from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set in environment variables")

llm = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)