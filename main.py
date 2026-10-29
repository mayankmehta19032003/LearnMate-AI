from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from llm_config import get_embeddings, get_llm
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

#----create embedding model for query-------
embedding_model = get_embeddings()

#-------load the vector db-------
vector_store = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embedding_model
)

#---------retriver (for more good search or retrive the ans from db)_-------
retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" :10,
        "lambda_mult" : 0.5
    }
)

#-----creating a model for ans it gives-----
llm = get_llm()

#-------prompt template--------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")



