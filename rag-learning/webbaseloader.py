from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

model=ChatHuggingFace
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-generation',
    temperature=0,
    huggingfacehub_api_token=token
)
model=ChatHuggingFace(llm=llm)
prompt=PromptTemplate(
    template=' Answer the following question \n {question} from the following text- \n {text}',
    input_variables=['question','text']
)
parser=StrOutputParser()
chain=prompt|model|parser

# loading web page to ask query 
url='https://www.amazon.in/ROMAND-Juicy-Lasting-Tint-PUMKIN/dp/B07ZKBDWCX/ref=asc_df_B07ZKBDWCX?mcid=94f864191bcd34bd8a5c0f32980981b6&tag=googleshopdes-21&linkCode=df0&hvadid=709857154056&hvpos=&hvnetw=g&hvrand=899431186696513408&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9302617&hvtargid=pla-1326403952593&hvocijid=899431186696513408-B07ZKBDWCX-&hvexpln=0&gad_source=1&th=1'
loader=WebBaseLoader(url)

docs=loader.load()

print(chain.invoke({'question':'what is the shade of the lip tint',  'text':docs[0].page_content}))

#python3 rag-learning/webbaseloader.py
