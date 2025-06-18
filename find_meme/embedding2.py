# use:
# --path : 欲embed文件的路径
# --increment : true-增量embed false-重置embed文件

from dotenv import load_dotenv
import os
import argparse
load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument('--path', type = str, nargs='+', default = [''], help = '欲embed图片的在src/output下的相对路径')
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


from dotenv import load_dotenv
import os
from openai import OpenAI  # 注意大小写


load_dotenv()  # 显式加载 .env 文件
API_KEY = os.getenv("INFINI_API_KEY")  # 必须与 .env 中的 KEY 完全一致（包括大小写）
BASE_URL = os.getenv("INFINI_BASE_URL")
 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

model_name = "deepseek-v3"

prompt_template='''
######
这里有一个短句，是罗翔老师（法律科普专家）的一句评价性的话，请用通俗的话语简要地重审他的话,
######要求：
1.不要凭空揣测句子的含义，只要将已有的句子解释，如果这句话未说完，也不要揣测接下来的内容。
例如“因为这里的问题是”这样不完整的句子，不要过多的揣测到问题的含义等等
2.请直接用通俗的话语重申语句即可
######句子：
{data}
######输出格式：
15字左右的通俗的重申，不要有任何其他内容
'''

def get_image_comment(data):
    prompt = prompt_template.format(data=data)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt},
                ],
            }
        ],
    )
    return completion.choices[0].message.content

documents = []
cnt = 0

for picture_path in picture_paths:
    for filepath, dirnames, filenames in os.walk(picture_path):
        for filename in filenames:
            # 获取图片注释
            comment = get_image_comment(filename)
            print(comment)
            # 创建带有注释的文档
            doc = Document(
                page_content=comment,  # 使用注释作为主要内容
                metadata={
                    'path': filepath,
                    'id': str(cnt),
                    'original_filename': filename  # 保留原始文件名
                }
            )
            documents.append(doc)
            cnt += 1

if args.reset:
    docsearch_chroma.reset_collection()
    print('embedding already reset')
print('===============================================') 
   
for i in range(0,len(documents),64):
    docsearch_chroma.add_documents(documents[i:i+64])
    print(f'successfully embed: {i}')