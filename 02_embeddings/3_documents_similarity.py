from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


embedding=HuggingFaceEmbeddings(model_name="openai/gpt-oss-20b",dimension=300)

documents=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership."
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills."
    "Sachin Tendulkar, also known as the "God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries."
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query='tell me about Virat kohli'

doc_embedding=embedding.embed_documents(documents)

query_embedding=embedding.embed_query(query)

#inside cosine similary both should be 2d list
scores=cosine_similarity([query_embedding],doc_embedding)[0]

index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print query
print(documents[index])
print("similarity score is ",score)

