# use:
# --path : 欲embed文件的路径
# --increment : true-增量embed false-重置embed文件

from dotenv import load_dotenv
import os
import argparse
load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument('--path', type = str, nargs='+', help = '欲embed图片的在src/output下的相对路径')
parser.add_argument('--reset', action='store_true', help = '选择--reset则并非增量embed')

args = parser.parse_args()

print(f'embedding picture directory names: {args.path}')
print(f'Reset embedding set: {args.reset}')
print('===============================================')

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
chroma_dir = "tmp/chroma_db"
docsearch_chroma = Chroma(
    embedding_function=baai_embedding,
    persist_directory=chroma_dir,
    collection_name="memes",
)

from langchain_core.documents import Document


picture_paths = []
for path in args.path:
    picture_paths.append(os.path.join('src/output', path))
print(f'embedding picture paths: {picture_paths}')

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

if args.reset:
    docsearch_chroma.reset_collection()
    print('embedding already reset')
print('===============================================') 
   
for i in range(0,len(documents),64):
    docsearch_chroma.add_documents(documents[i:i+64])
    print(f'successfully embed: {i}')