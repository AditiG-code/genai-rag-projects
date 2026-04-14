from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

model=ChatHuggingFace

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=200,
    huggingfacehub_api_token=token
)

model=ChatHuggingFace(llm=llm)

chathistory=[
    SystemMessage(content="You are a helpfull assistant")
]

while True:
    user_input=input('You:')
    chathistory.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    result=model.invoke(chathistory)
    chathistory.append(AIMessage(content=result.content))

    print("AI :",result.content)


#python3 projects/minichatbot.py
    