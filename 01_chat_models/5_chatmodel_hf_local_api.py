from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm=HuggingFacePipeline.from_model_id(
    model_id="openai/gpt-oss-20b",
    task="text_generation",
    pipeline_kwards=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model=ChatHuggingFace(llm=llm)
result=model.invoke("Who is Hermione Granger")
print(result.content)
#python3 Chatmodels/5_chatmodel_hf_local_api.py