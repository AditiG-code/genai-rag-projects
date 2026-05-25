from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableBranch,RunnableSequence

import os
load_dotenv()
model=ChatHuggingFace
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text_generation",
    temperature=0.5,
    huggingfacehub_api_token=token
)
model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='write a detail report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarize the given {text}',
    input_variables=['text']
)

# topic->model->report if more than 500 words then summarize
#  else print report as it is

report_gen=RunnableSequence(prompt1 | model| parser)

branch_chain=RunnableBranch(
    # if (condition,runnable)
    # default
    # x is output from the parser obtained after running report_gen chain 
    (lambda x: len(x.split())>=100,RunnableSequence(prompt2|model|parser) ),
    # else run default condition
    RunnablePassthrough()
)

# now these two chain will combine to give desired output
final_chain=RunnableSequence(report_gen,branch_chain)

print(final_chain.invoke({'topic' : " Cricket team India vs Australia"}))

#python3 06_runnables/runnable_branch.py


