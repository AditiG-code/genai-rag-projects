from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
model=ChatHuggingFace

llm1=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text_generation",
    temperature=0,
    huggingfacehub_api_token=token
)
llm2=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text_generation",
    temperature=0,
    huggingfacehub_api_token=token
)

model1=ChatHuggingFace(llm=llm1)

model2=ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser
chain = parallel_chain | merge_chain



text='''
India has been a federal republic since 1950. Its government is a democratic parliamentary system. It is a multilingual (multiple languages) and multicultural (multiple cultures) society.[36] The capital city of India is New Delhi. India has the second largest military force in the world and is also a nuclear weapon state.[37] India's economy became the world's fastest growing in the G20 developing nations during 2014, replacing the People's Republic of China.[38] India's literacy and wealth are also rising.[39]

India has the fourth largest economy by nominal GDP, the third largest by GDP (PPP) and is one of the fastest growing major economy. According to New World Wealth, India is the fifth richest country in the world with a total individual wealth of $12.6 trillion.[40][41] However, it still has many social and economic issues, for example poverty, pollution, social equality, religious extremism, terrorism and corruption.[42] India has reduced its rate of poverty but its economic inequality has increased.[43]

India is a founding member of the World Trade Organisation (WTO), and has signed the Kyoto Protocol. It is also a member of the G20 developing nations. India has its own space agency (ISRO). It has done much research throughout the Solar System. It has sent spacecraft to the Moon and Mars. Indian movies, music and spiritual teachings are becoming more important in global culture.[44] Sources describe it as a potential superpower, because of its rising economy and increase in global influence. India is a country with nuclear weapons. It also has a high rank in military expenditure. It has disputes over Kashmir with its neighbours, Pakistan and China, since the middle of the 20th century.[45]

India has the fourth largest number of spoken languages per country in the world, only behind Papua New Guinea, Indonesia, and Nigeria.[46] Most of Indians follow Hinduism at 80%, but people of different religions such as Buddhism, Sikhism and Islam also live there
'''
result=chain.invoke({'text':text})

print(result)
#chain.get_graph().print_ascii()

#python3 04_chains/parallel.py