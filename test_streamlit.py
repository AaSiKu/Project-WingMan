import streamlit as st
import os
import os
import json
from llama_index.core import SimpleDirectoryReader
import ollama
from groq import Groq
from langchain import PromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from pathlib import Path
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pickle
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
GEMINI_API_KEY = os.environ.get("GEMINI_API")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=GEMINI_API_KEY)

# Create a Chroma instance
vector_store = Chroma(embedding_function=embeddings, persist_directory="Summaries-2")


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size = 15,  # Controls the maximum burst size.
)
GEMINI_API_KEY=os.environ.get("GEMINI_API")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# question = "Explain me what is the attention mechanism in 100 words"
# context = retriever.invoke(question)
# vector_store = FAISS(index_path="data/vector_store")
retriever_or = vector_store.as_retriever()
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=GEMINI_API_KEY,rate_limiter=rate_limiter)

def stream_summary_response(question):
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=GEMINI_API_KEY,rate_limiter=rate_limiter)
    chunks_prompt="""
    Using the Context bellow answer the question {question}, mention the path of the MOST RELAVENT documents.

    NOTE: The contexts are summaries of maybe very large documents, so scrutinize it well and at the end of each summary keywords are also mentioned, use these also to answer the question.RETURN ONLY PATHS TO THE MOST RELEVANT DOCUMENTS
    Context:
    {text}

    IMPORTANT: The answer should be in the following format:
    RETURN ONLY JSON DATA NOTHING ELSE
    ```
        {{
        "files": [
            {{
            "file_path": "path to the file ",
            "reason":"How is the file relevant to the question"
            }}
        ]
        }}
        ```
    """
    RAG_prompt_template=PromptTemplate(input_variables=['text','question'],
                                        template=chunks_prompt)
    rag_chain = (
    {"text": retriever_or | format_docs, "question": RunnablePassthrough()}
    | RAG_prompt_template
    | gemini_llm
    | StrOutputParser()
)
    response = ""
    for s in rag_chain.stream(question):
        # st.write(s,end=' ')
        response += s
    # return response
    formated_response = response.strip('`').split('json')[1]
    json_response = json.loads(formated_response)
    json_response
    File_paths_relavent = [x['file_path'] for x in json_response['files']]
    st.write("Relevant Paths Found:\n\n",File_paths_relavent)

    return File_paths_relavent
# question = "What is exergy?"

# response = ""
# for s in rag_chain.stream(question):
#     print(s)
#     response += s
# response

def input_llm_embeddings_pdf(File_path): #!Pass the file path of the pdf, this function will return the split_chunks
    # Reading the pdf file
    pdfreader = PdfReader(File_path)
    text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            text += content
    # Converting the text of the pdf of Document object
    docs = [Document(page_content=text)]
    docs
    ## Splittting the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200) # 300k characters per chunk, or nearly 75,000 tokens
    chunks = text_splitter.create_documents([text])
    print(chunks)
    split_chunks = []
    for x in range(0,len(chunks),1400):
        split_chunks.append(chunks[x:x+1400])
    return split_chunks

retriever = vector_store.as_retriever()
chunks_prompt="""
You are a helpful teacher, you have to answer the question asked by the student.
Using the Context bellow answer the question {question}, also mention the page number, name of the doc, etc relavent details at the end of the response.
Context:
{text}
Answer in as much detail as possible, if no direct response try to infer from the context.If cannot answer return "Cannot infer from the context",along with what you infered from the context.Format the response well 
"""
RAG_prompt_template=PromptTemplate(input_variables=['text','question'],
                                    template=chunks_prompt)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_google_genai import ChatGoogleGenerativeAI
GEMINI_API_KEY=os.environ.get("GEMINI_API")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=GEMINI_API_KEY)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]=os.environ.get("LANGCHAIN_API")
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="LearnLang"
# Assuming rag_chain is already defined and set up
# Example:
# rag_chain = YourChainObject()

def stream_question(question, File_paths_relavent):
    index = 0
    rag_chain = (
    {"text": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_prompt_template
    | gemini_llm
    | StrOutputParser())
    path_vector_database = Path('VectorDatabases')
    while index < len(File_paths_relavent):
        file_path = File_paths_relavent[index]
        print(f"Processing file: {file_path}")
        print(f"Index {index+1} of {len(File_paths_relavent)} completed")

        path_vector_db_folder = Path.joinpath(path_vector_database,file_path)
        # path_vector_db_folder.mkdir(parents)
        path_vector_db_folder.mkdir(parents=True, exist_ok=True)
        if not path_vector_db_folder.exists():
            split_chunks = input_llm_embeddings_pdf(file_path)
            try:
                for chunks in split_chunks:
                    vector_store = Chroma(embedding_function=embeddings, persist_directory=str(path_vector_db_folder))
                    # print(f"Adding document {docs}")
                    vector_store.add_documents(chunks)
            except Exception as e:
                print(f"Exception: {e}")
                print("Resource exhausted, waiting for 60 seconds...")
                index -= 1
                time.sleep(60)
        index += 1
    # Use an empty container to update the output
    output = st.empty()
    # Stream each result from the chain
    result = ""
    st.write_stream(rag_chain.stream(question))
    # for s in rag_chain.stream(question):
    #     # Print each result in the Streamlit app
    #     result += s
        # st.empty()
        # st.text("")
        # st.markdown(result)  # You can use st.text(s) or st.markdown(s) for different formatting

# Streamlit UI
st.title("Streaming Question Answering")

# Input question from user
question = st.text_input("Enter your question:")
st.subheader("Following are the paths of the most relevant documents, gotten from summaries")
relavent_paths = stream_summary_response(question)
st.write("Processing the question...")
st.subheader("Results:")
stream_question(question, relavent_paths)
# # When a question is provided, stream the output
# if question:
#     st.write("Processing the question...")
#     stream_question(question, relavent_paths)
