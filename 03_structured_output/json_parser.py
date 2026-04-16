'''
First given input to llm 
then detailed report as output
again given this output as input to llm 
and then want output as a 5 line summary of data
'''

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    #partial means it will run during runtime
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'mount everest'})

print(result)

#python3 03_structured_output/json_parser.py
