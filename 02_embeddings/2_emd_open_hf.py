from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name="openai/gpt-oss-20b")

text="Delhi is the capital of India"

vector=embedding.embed_query(text)
print(str(vector))