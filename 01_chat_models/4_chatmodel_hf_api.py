from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
model=ChatHuggingFace
#create a variable to tell which model u want to use
llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text_generation",
    temperature=1.5,
    huggingfacehub_api_token=token

)

model=ChatHuggingFace(llm=llm)
result=model.invoke("who is hermione granger?")
print(result.content)
#python3 Chatmodels/4_chatmodel_hf_api.py