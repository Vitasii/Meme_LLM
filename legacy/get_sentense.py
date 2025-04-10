from dotenv import load_dotenv  
import os  

load_dotenv()

CHAT_MODEL="deepseek-r1"

os.environ["OPENAI_API_KEY"]=os.environ.get("INFINI_API_KEY")  # langchain use this environment variable to find the OpenAI API key
OPENAI_BASE=os.environ.get("INFINI_BASE_URL")

from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(
    temperature=0, 
    model=CHAT_MODEL,
    base_url=OPENAI_BASE)

def print_with_type(res):
    print(f"%s" % (type(res)))
    print(f"%s" % res)

from langchain.prompts.chat import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个表情包文案专家，根据用户描述和名人特征生成幽默短句。！！！注意，你生成的文字是需要作为表情包内容的，请反思你输出的内容是否符合表情包的需求"),
    ("human", "用户需求：{user_input}\n关联名人：{celebrity}（特征：{description}）")
])

chat_model = ChatOpenAI(
    temperature=0, 
    model=CHAT_MODEL,
    base_url=OPENAI_BASE)

chain = chat_prompt | chat_model

result = chain.invoke({
    "user_input": "表达惊讶",
    "celebrity": "张维为",
    "description": "知名学者，标志性笑容"
})
print_with_type(result)