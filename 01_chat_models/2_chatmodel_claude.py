from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv() #this hits the API key 
model=ChatAnthropic(model='claude-3-5-sonet-20241022')
result=model.invoke("What is the capital of India?")
print(result)