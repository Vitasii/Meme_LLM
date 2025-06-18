import gradio as gr
from PIL import Image
import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHAT_MODEL="deepseek-v3"
openai_api_key = os.environ.get("INFINITE_API_KEY")
openai_base_url = os.environ.get("INFINITE_BASE_URL")

client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

def get_response(prompt): 
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt},]
    )
    return response.choices[0].message.content

baai_embedding = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url=os.environ.get("SF_BASE_URL"),
    api_key=os.environ.get("SF_API_KEY"),
)

chroma_dir = "tmp/chroma_db"
docsearch_chroma = Chroma(
    embedding_function=baai_embedding,
    persist_directory=chroma_dir,
    collection_name="memes",
)

def get_image(img_path):
    if not os.path.exists(img_path):
        return None  
    try:
        with Image.open(img_path) as img:
            return img.copy()  
    except Exception as e:
        return None  
    
def show_results(docs, image_list):
    for result in docs:
        print(result)
        cur_path = os.path.join(result.metadata['path'], result.page_content)
        image_list.append(get_image(cur_path))



def find_meme(query):

    prompt = f"""
    你是一个文字生成器。我将会给你一段文本，你需要根据它生成一些与这段文本相关的句子。你可以使用以下方法：
    
    1. 你可以就文本中出现的概念进行进一步解释，或者给出文本中出现的事物的更多细节。你可以进行网络搜索。 
    
    2. 你可以给出或分析文本中提到的事物产生的原因；你可以从若干个不同的角度、立场分析原因。
    
    3. 你可以给出一些应对文本中提到的事物的对策。你可以从若干个不同的事物参与主体出发，给出可能得应对策略。
    
    4. 你可以就一些文本中提到的观点或者事物进行评论或者评价。你可以从若干个不同的角度、立场（如支持、反对、中立）进行评价。
    
    ########## 
    
    文本：{query}
    
    ##########
    
    格式要求：
    
    1. 仅返回一系列句子，一句一行。
    
    2. 句子的总数量大概在10左右。
    """

    response = get_response(prompt)
    responses = response.split('\n')
    print(responses)

    image_list = []
    for text in responses:
        docs = docsearch_chroma.similarity_search(text, k=1)
        show_results(docs, image_list)
        
    return image_list


demo = gr.Interface(
    fn=find_meme,
    inputs=gr.Textbox(label="query"), 
    outputs=gr.Gallery(label="图片"),
)

demo.launch()