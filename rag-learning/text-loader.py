from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
import os

model=ChatHuggingFace
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    temperature=0,
    huggingfacehub_api_token=token
)
model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

prompt=PromptTemplate(
    template='Summarize in 5 lines {text}',
    input_variables=['text']
)

loader=TextLoader('/Users/aditigupta/langchain-RAG/rag-learning/india.txt',encoding='utf-8')

docs=loader.load()

chain=prompt |model|parser

print(chain.invoke({'text':docs[0].page_content}))

# python3 rag-learning/text-loader.py