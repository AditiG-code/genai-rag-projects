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
model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Generate detailed report on the {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Write 5 facts about given topic./n {text}',
    input_variables=['text']
)

parser=StrOutputParser()
chain =prompt1 | model | parser |prompt2 |model |parser

result=chain.invoke({'topic':'Mount Everest'})
#print(result)

chain.get_graph().print_ascii()

#python3 04_chains/sequential.py