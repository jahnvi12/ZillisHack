from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OCTOAI_API_TOKEN"] = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjNkMjMzOTQ5In0.eyJzdWIiOiI2Yzg0MjMyNy02ZmY4LTRkMzYtODNhNi02NWRjYjZiMDVjNjIiLCJ0eXBlIjoidXNlckFjY2Vzc1Rva2VuIiwidGVuYW50SWQiOiI2NjA4NmQzZS1iNWQyLTQxOTgtOGM0MS1hYWZjMmQ1MDAxM2UiLCJ1c2VySWQiOiI5OGQ3MDRjZC1hOWViLTQ4MjktOTQxZS0yZDA3N2VhYjU4NjMiLCJyb2xlcyI6WyJGRVRDSC1ST0xFUy1CWS1BUEkiXSwicGVybWlzc2lvbnMiOlsiRkVUQ0gtUEVSTUlTU0lPTlMtQlktQVBJIl0sImF1ZCI6IjNkMjMzOTQ5LWEyZmItNGFiMC1iN2VjLTQ2ZjYyNTVjNTEwZSIsImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkub2N0b21sLmFpIiwiaWF0IjoxNzExMjEzMDUxfQ.ToHPKaVUUGtAFtKcqja-FhJxJiXNiggPkZ1ZsbNk0evngkw0m1Nes5hdt4hZF4Q7KqHa3kxAAmnwP-VUiW2RnByh6g0WOQWeffxIjwtbL17hGXQt3nJSs3XDfgluBKVyfcCeIoDjz56uciMeBSmFHrrh7asbtmij7CRDyn0DRfmZBEI7y6Psv8Tu2vwSmlyeUxzSqmu5iMNsRalndbeQaKOTQAtGtnp8zPVogivYc_L26iFApUg6EN01KU0GRGg8KdH6PgPF6sIuc5o5LsPHImKeFumgE9XO6W9GK0dYofQE2zLek1IRNhuQw25wDHm7Rmd54_Y0WH1OCqBGExGTBg"

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n Instruction:\n{question}\n Response: """
prompt = PromptTemplate.from_template(template)

llm = OctoAIEndpoint(
    endpoint_url="https://text.octoai.run/v1/chat/completions",
    model_kwargs={
        "model": "mixtral-8x7b-instruct-fp16",
        "max_tokens": 128,
        "presence_penalty": 0,
        "temperature": 0.01,
        "top_p": 0.9,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Keep your responses limited to one short paragraph if possible.",
            },
        ],
    },
)

loader = PyPDFLoader('./data/Kanishka.pdf')
documents = loader.load_and_split()

# print(documents)

embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")

vector_store = Milvus.from_documents(
    documents,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": 19530},
    collection_name="menus"
)

retriever = vector_store.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "How much does chicken 65 cost in Kanishka?"

print(chain.invoke(question))