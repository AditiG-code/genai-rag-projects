'''
First given input to llm 
then detailed report as output
again given this output as input to llm 
and then want output as a 5 line summary of data
'''

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

#using chain and parser in this
parser=StrOutputParser()

'''
made chain 
temlate :input1 -> model ->output1 ->parser (generated structure output )->this as input again ->model
->parser(generated summar ) ->finalresult
'''
chain=template1 | model | parser | template2 | model | parser 

result=chain.invoke({'topic':'hermione granger'})
print(result)

#python3 03_structured_output/output_using_parser.py