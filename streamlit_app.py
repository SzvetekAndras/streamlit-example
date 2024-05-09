import streamlit as st
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
from langchain_community.vectorstores import Chroma
import pandas as pd
import re
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ae-oa-d-we-004.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "7d38d7a887484ee4b289d0b65e0b31a5"


    
def get_vectorstore(embeddings, documents=None, persist_path=None):
    if persist_path is not None and os.path.exists(persist_path):
        return Chroma(persist_directory=persist_path, embedding_function=embeddings)
    else:
        return Chroma.from_documents(documents, embeddings, persist_directory=persist_path)

def search(vectorstore, query, top_k=5):
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    df = pd.DataFrame(results, columns=["Document", "Score"])
   # df["Document"] = df["Document"].str.replace(r"page_content='nan' metadata={", "\n")
    return df.to_string(index=False)
    
if __name__ == "__main__":

    #text-embedding-ada-002
    embeddings = AzureOpenAIEmbeddings(
    azure_deployment="me-emb-ada-002",
)
    
    vectorstore = Chroma(persist_directory='./vectorstore', embedding_function=embeddings)
    st.title("Requirement Searchbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response = search(vectorstore, prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
