import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

st.title("Chatbot Interface")
st.write("Welcome to the Chatbot! Please enter your message below.")

with st.sidebar:
    st.title("Q&A Chat")
    st.write("Welcome User.")
    st.markdown(
        """
        ## About
        This app help you to upload a document and get an answer based upon the uploaded content.
        
        It is using:
        - [LangChain](https://python.langchain.com/)
        - [Streamlit](https://streamlit.io/)
        - [OpenAI](https://platform.openai.com/docs/models) ->   LLM Model
        - For Additional Information please contact. Aakash Jha
        """
    )
    add_vertical_space(5)
    st.write("This is some more content in the sidebar.")
    
    
    
def main():
    st.header(" Chat with PDF ")
    
    load_dotenv()
    
       # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        st.write("PDF File Name is:", pdf.name)
    else:
        st.write("No file uploaded yet. Please upload a PDF file.")
    
    #st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write("Number of pages in the PDF:", len(pdf_reader.pages))
        #st.write(pdf_reader)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #Create a text splitter, with a chunk size of 1000 characters and an overlap of 200 characters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks)
        
        #Embedding  -Running at multiple time do have a cost associated with it
        
        store_name =pdf.name[:-4]
        
        if os.path.exists(f"{store_name}/index.faiss"):  # Check if FAISS index file exists
             embedding = OpenAIEmbeddings()
            # VectorStore = FAISS.load_local(f"{store_name}", embedding)
             VectorStore = FAISS.load_local(f"{store_name}", embedding, allow_dangerous_deserialization=True)

             st.write("Embedding loaded from disk")
        else:
             embedding = OpenAIEmbeddings()
             VectorStore = FAISS.from_texts(chunks, embedding)
             VectorStore.save_local(f"{store_name}")
             
             st.write("Embedding created and saved to disk")
    
           
        #Accept user question/ query
        query = st.text_input("Ask a question about the PDF:")
        if query:
            #st.write("Query:", query)
            #st.write("VectorStore:", VectorStore)
            docs = VectorStore.similarity_search(query=query, k=3)  #returns top 3 docs object from the vector store
           # st.write(docs)   #returnts top 3 docs object from the vector store
           #cost with default LLM Model is $0.01714
           #cost with gpt-3.5-turbo is $0.001566
            llm =OpenAI()  #if needed we can chnage the LLM Model , currently it is using default LLM Model devinci
            #llm=OpenAI(model_name='gpt-3.5-turbo')
            chain=load_qa_chain(llm=llm, chain_type="stuff")
            #cost
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)            
            
        else:
            st.write("Please enter anothe question to get an answer. Answer for current question is not available.")
                
       # st.write("Extracted text from the PDF:")
       # st.write(text)
    
    
 #To run the code: streamlit run app.py
 #To install the deoendencies: pip3 install -r requirements.txt   
    
if __name__== '__main__':
     main()   

