import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
from langchain_community.vectorstores import Chroma
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ae-oa-d-we-004.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "7d38d7a887484ee4b289d0b65e0b31a5"

def get_dataloader(excel_files):
    assert len(excel_files) > 0
    df = pd.read_excel(excel_files.pop())

    while len(excel_files) > 0:
        df = pd.concat([df, pd.read_excel(excel_files.pop())])
    
    df = df[df['DA_Object_Type']=='Requirement'][['ID', 'Object Text']]
    
    return DataFrameLoader(df, page_content_column="Object Text")
    
def get_vectorstore(embeddings, documents=None, persist_path=None):
    if persist_path is not None and os.path.exists(persist_path):
        return Chroma(persist_directory=persist_path, embedding_function=embeddings)
    else:
        return Chroma.from_documents(documents, embeddings, persist_directory=persist_path)
    
def main():
    excel_files = ['streamlit-example/req/SWRS_SIT.xlsx', 'streamlit-example/req/SysRS.xlsx']
    loader = get_dataloader(excel_files)
    documents = loader.load()

    #text-embedding-ada-002
    embeddings = AzureOpenAIEmbeddings(
   azure_deployment="me-emb-ada-002",
)
    text = "this is a test document"
    query_result = embeddings.embed_query(text)
    doc_result = embeddings.embed_documents([text])
    doc_result[0][:5]
    
    vectorstore = get_vectorstore(embeddings, documents, persist_path='./vectorstore')
   # q = 'Under what conditions related to the position of a target vehicle relative to the subject vehicle should the Blind Spot Detection/Lane Change Assist (BSD/LCA) warning level remain at 0?'
    print("Vectorstore available")
    while True:
        print("Type 'exit' to quit")
        q = input("Enter a query: ")
        if q == 'exit':
            break
        else:
            res = search(vectorstore, q)
            print(res)

def search(vectorstore, query, top_k=5):
    return vectorstore.similarity_search_with_score(query, k=top_k)


if __name__ == "__main__":
    main()