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

# prompt for judgement
def return_prompt(query, meme_name):
    
    prompt = f"""现在有一个希望找到一个meme图的询问和一个回答（以meme的文字内容为形式给出）。
    请根据以下规则来评估这个回答。
    
    下面将给出若干个维度，你需要在这些维度上给表情包打分。分数是一个0到20之间的正整数（0代表最差，20代表最好，10分代表一般）：

    1. 相关性: 这个回答对于询问的契合度有多高？这种契合可能需要一定的前提，如回应者的身份，或者出于某种立场。
    2. 趣味性: 这个回答是否足够新颖？是否幽默？这个回答甚至可以具有一定的冒犯性。
    3. 针对性: 这个回答是否是对于询问的场景下的针对性的回答？我们不希望回答的泛用性过强，导致可以回答很多种不同的询问

    对于每个维度，给出一个评分以及简短的解释。
    
    ############
    
    meme图的询问： {query}
    
    回答：{meme_name}
    
    ############
    
    格式要求：你需要按以下个格式来格式化你的输出。
    {{
        "相关性": {{
        "score": <score>,
        "explanation": "<explanation>"
        }},
        "趣味性 ": {{
        "score": <score>,
        "explanation": "<explanation>"
        }},
        "针对性": {{
        "score": <score>,
        "explanation": "<explanation>"
        }}
    }}
    """

    return prompt

# copied from lab6
def my_parse_judge_output(judge_output):
    # this inherited function is widely used in research repositories
    # it does some simple fixes to the output to make it a valid JSON
    try:
        # 尝试提取花括号包裹的部分
        match = re.search(r"\{.*\}", judge_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the output.")
        
        json_str = match.group(0)

        # 简单修复常见 JSON 错误：缺逗号、缺括号等
        json_str = re.sub(r'("explanation"\s*:\s*"[^"]+)"\s*(")', r'\1,\2', json_str)  # 补逗号
        json_str = json_str.strip()

        # 尝试解析 JSON
        return json.loads(json_str)

    except Exception as e:
        print(f"Error parsing judge output: {e}")
        print(f"Raw output: {judge_output}")
        return {
            "相关性": {"score": 10, "explanation": "Failed to parse"},
            "趣味性": {"score": 10, "explanation": "Failed to parse"},
            "针对性": {"score": 10, "explanation": "Failed to parse"}
        }
     
def judge_meme(candidates, k, query):
    can_and_score = []
    for name in candidates:
        judgement = my_parse_judge_output(get_response(return_prompt(query, name[1]), CHAT_MODEL2))
        print(f"judgement of {name[1]}:")
        print(judgement)
        sum_score = 1 * judgement['相关性']['score'] 
        sum_score += 1 * judgement['趣味性']['score'] 
        sum_score += 1 * judgement['针对性']['score']
        can_and_score.append((name, sum_score))
        
    sorted_candidates = sorted(can_and_score, key=lambda pair: pair[1], reverse=True)
    print(sorted_candidates)
    select_list = []
    for i in range(min(k, len(sorted_candidates))):
        select_list.append(sorted_candidates[i][0])
        
    return select_list
   

def find_meme(query, num_meme):

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

    num_candidate = math.ceil(num_meme / 2)
    
    response = get_response(prompt, CHAT_MODEL)
    responses = response.split('\n')
    print("text responses generated:")
    print(responses)
    print("===================================")

    image_list = []
    candidate_names = []
    for text in responses:
        docs = docsearch_chroma.similarity_search(text, k=num_candidate)
        for result in docs:
            print(result)
            candidate_names.append((result.metadata['path'], result.page_content))
    print("meme candidates:")
    print(candidate_names)
    print("===================================")
    
    meme_list = judge_meme(candidate_names, num_meme, query)
    print("===================================")
    print("meme selected:")
    print(meme_list)
    print("===================================")
    image_list = []
    for meme in meme_list:
        image_path = os.path.join(meme[0], meme[1])
        image_list.append(get_image(image_path))
        
    return image_list


demo = gr.Interface(
    fn=find_meme,
    inputs=[gr.Textbox(label="query"),  
            gr.Slider(minimum=1, maximum=10, step=1, label="# of generated memes")], 
    outputs=gr.Gallery(label="generated memes"),
)

demo.launch()