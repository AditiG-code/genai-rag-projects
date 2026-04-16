from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
model=ChatHuggingFace

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text_generation",
    temperature=0,
    huggingfacehub_api_token=token
)
prompt=PromptTemplate(
    template='Generate 5 facts about {topic}',
    input_variables=['topic']

)
model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

chain=prompt|model|parser
result=chain.invoke({'topic':'Hermione Granger'})
print(result)

#python3 04_chains/simple.py
