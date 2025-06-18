import gradio as gr
from PIL import Image
import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import re, time, json, math, threading, random
from openai import RateLimitError

load_dotenv()

CHAT_MODEL="deepseek-v3"
CHAT_MODEL2="llama-3.3-70b-instruct"
CHAT_MODEL3 ="deepseek-r1"

openai_api_key = os.environ.get("INFINITE_API_KEY")
openai_base_url = os.environ.get("INFINITE_BASE_URL")

client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

def get_response(prompt, model_name): 
    response = client.chat.completions.create(
        model=model_name,
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

# MODIFIED: 新增full_content参数并整合到prompt

def my_parse_judge_output(judge_output):
    try:
        match = re.search(r"\{.*\}", judge_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the output.")
        
        json_str = match.group(0)
        json_str = re.sub(r'("explanation"\s*:\s*"[^"]+)"\s*(")', r'\1,\2', json_str)
        json_str = json_str.strip()
        return json.loads(json_str)

    except Exception as e:
        print(f"Error parsing judge output: {e}")
        print(f"Raw output: {judge_output}")
        return {
            "相关性": {"score": 10, "explanation": "Failed to parse"},
            "趣味性": {"score": 10, "explanation": "Failed to parse"}
        }
     
# MODIFIED: 处理新的三元组数据结构 (path, name, content)
def judge_meme(candidates, k, query):
    # 构建批量评估prompt
    batch_prompt = f"""请从以下候选回答中选择出对于回答问题【{query}】最合适，或者语义与其最相近的{k}个。

候选回答列表：
"""
    
    # 添加所有候选信息
    for idx, item in enumerate(candidates):
        path, name = item[0], item[1]
        content = item[2] if len(item) > 2 else ""
        batch_prompt += f"\n{idx+1}. {name}"
        if content:
            batch_prompt += f"\n   描述: {content[:100]}{'...' if len(content)>100 else ''}"
    
    batch_prompt += f"""
注意：以上可能有相同的语句，这样相同的语句你只能选一次
请直接返回最合适的{k}个编号，按匹配度从高到低排列，格式为：
最佳选择: 编号, 编号, ..., 编号
例如：最佳选择: 3, 1, 5
"""
    
    # 获取AI批量选择结果
    selection_response = get_response(batch_prompt, CHAT_MODEL)
    print(f"AI批量选择结果: {selection_response}")
    
    # 解析选择结果
    selected_ids = []
    if "最佳选择:" in selection_response:
        try:
            selected_part = selection_response.split("最佳选择:")[1].strip()
            selected_ids = [int(num.strip()) for num in selected_part.split(",") if num.strip().isdigit()]
        except Exception as e:
            print(f"解析选择结果出错: {e}")
            selected_ids = list(range(min(k, len(candidates))))
    
    # 默认回退：如果解析失败，取前k个
    if not selected_ids:
        selected_ids = list(range(min(k, len(candidates))))
    
    # 返回选中的候选
    return [candidates[i-1] for i in selected_ids if 0 < i <= len(candidates)][:k]


   
def find_meme(query, num_meme):
    prompt = f"""
    你需要对将要提供的一段文本做出回应。可以参考以下若干种回应方式：
    
    1. 你可以就文本中出现的概念进行进一步解释，或者给出文本中出现的事物的更多细节。你可以进行网络搜索。 
    
    2. 你可以给出或分析文本中提到的事物产生的原因；你可以从若干个不同的角度、立场分析原因。
    
    3. 你可以给出一些应对文本中提到的事物的对策。你可以从若干个不同的事物参与主体出发，给出可能得应对策略。
    
    4. 你可以就一些文本中提到的观点或者事物进行评论或者评价。你可以从若干个不同的角度、立场（如支持、反对、中立）进行评价。
   
    ##########
    
    文本：{query}
    
    ##########
    
    内容要求：
    
    1.一共提15句，角度不同，第一句很直接地表达看法，例如（“不能”，“愚蠢”），其余句子以任意方式回应。
    
    2.无论如何，应该用比较通俗的话语来表达，而且句子要简短，最好15字左右。
    
    ##########
    
    格式要求：
    
    将你想要的15句话转为用通俗的语言输出，只输出这15句话，一行一句，除此之外不要有任何内容

    """

    num_candidate = math.ceil(num_meme)
    
    response = get_response(prompt, CHAT_MODEL)
    responses = response.split('\n')
    responses = [s.strip() for s in responses if s.strip()]
    print("text responses generated:")
    print(responses)
    print("===================================")

    candidate_names = []
    for text in responses:
        docs = docsearch_chroma.similarity_search(text, k=num_candidate)
        for result in docs:
            # MODIFIED: 存储三元组 (path, original_filename, page_content)
            candidate_names.append((
                result.metadata['path'],
                result.metadata.get('original_filename', result.page_content.split('\n')[0]),
                result.page_content,
                result.metadata['id']
            ))
    print("meme candidates:")
    for c in candidate_names: 
        print(c)
    print("===================================")
    
    meme_list = judge_meme(candidate_names, num_meme, query)
    print("===================================")
    print("meme selected:")
    for c in meme_list: 
        print(c)
    print("===================================")
    
    image_list = []
    for meme in meme_list:
        # MODIFIED: meme[1]始终是原始文件名
        image_path = os.path.join(meme[0], meme[1])
        img = get_image(image_path)
        if img:
            # MODIFIED: 添加注释到图片元数据
            img.info['description'] = meme[2] if len(meme) > 2 else ""
        image_list.append(img)
        
    return image_list

# MODIFIED: 增强Gallery显示配置
demo = gr.Interface(
    fn=find_meme,
    inputs=[gr.Textbox(label="query"),  
            gr.Slider(minimum=1, maximum=10, step=1, label="# of generated memes")], 
    outputs=gr.Gallery(label="generated memes"),
)


if __name__ == "__main__":
    demo.launch(share=True)
