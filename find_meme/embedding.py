from dotenv import load_dotenv
import os
load_dotenv()

CHAT_MODEL="deepseek-v3"
os.environ["OPENAI_API_KEY"]=os.environ.get("INFINITE_API_KEY") 
os.environ["OPENAI_BASE_URL"]=os.environ.get("INFINITE_BASE_URL")

from langchain_openai import OpenAIEmbeddings
baai_embedding = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url=os.environ.get("SF_BASE_URL"),
    api_key=os.environ.get("SF_API_KEY"),
)

from langchain_chroma import Chroma
chroma_dir = "/scratch1/chroma_db"
docsearch_chroma = Chroma(
    embedding_function=baai_embedding,
    persist_directory=chroma_dir,
    collection_name="memes",
)

from langchain_core.documents import Document

picture_paths = ["../src/output/vv",]

titles = []
paths = []
documents = []
cnt = 0


for picture_path in picture_paths:
    for filepath, dirnames, filenames in os.walk(picture_path):
        for filename in filenames:
            titles.append(filename)
            paths.append(filepath)
            doc = Document(page_content=filename, metadata={'path': filepath, 'id': str(cnt)})
            documents.append(doc)
            # print(filename)
            cnt += 1

docsearch_chroma.reset_collection()
for i in range(0,len(documents),64):
    docsearch_chroma.add_documents(documents[i:i+64])