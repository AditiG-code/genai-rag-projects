'''
First given input to llm 
then detailed report as output
again given this output as input to llm 
and then want output as a 5 line summary of data
'''

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
model=ChatHuggingFace

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text_generation",
    temperature=1.5,
    huggingfacehub_api_token=token
)
model=ChatHuggingFace(llm=llm)

#1 input to output
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

#2 output as input
template2=PromptTemplate(
    template='Write 5 line summary on the given text. /n {text}',
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'hermione granger'})

result=model.invoke(prompt1)

prompt2=template2.invoke({'text':result.content})

finalresult=model.invoke(prompt2)

print(finalresult.content)


#python3 03_structured_output/output_without_parser.py
