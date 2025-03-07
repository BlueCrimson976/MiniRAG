### Author : Rishita Ray
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import streamlit as st 
from langchain_core.messages import HumanMessage , AIMessage
import PyPDF2 
import torch
from pathlib import Path
import os
import fitz
from PIL import Image
import pymupdf
import io
from streamlit.web import cli as stcli
from streamlit import runtime
import sys

torch.classes.__path__ = []

st.set_page_config(
        page_title='Minimalistic RAG',
        page_icon= "🔍"
    )

save_folder = 'uploaded_references'
Path(save_folder).mkdir(parents=True, exist_ok=True)        

@st.cache_data
def load_doc(filepdf):
    pdf_reader = PyPDF2.PdfReader(filepdf)
    return pdf_reader


@st.cache_data
def img_loader(img_file):
  img_file.seek(0,0)
  doc = fitz.open(stream=img_file.read(), filetype="pdf")
  for page_index in range(len(doc)):
      page = doc.load_page(page_index)
      image_list = page.get_images(full=True)
      if image_list:
        print(f"[+] Found a total of {len(image_list)} images on page {page_index}")
      else:
        print("[!] No images found on page", page_index) 
      for image_index, img in enumerate(image_list, start=1):
                xref = img[0]
                image = doc.extract_image(xref)
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"image{page_index+1}_{image_index}.{image_ext}"
                with open(save_folder+'\\'+image_name, "wb") as image_file:
                     image_file.write(image_bytes)
  st.success('All pdf images extracted and loaded uploaded_references folder.')

@st.cache_data
def streamImage(save_folder):
     image_list = os.listdir(save_folder)
     for i in image_list:
         d = save_folder+'\\'+i
         st.session_state.clicked == True
         st.image(save_folder+'\\'+i) 


def clear_gallary(save_folder):
    image_list = os.listdir(save_folder)
    for i in image_list:
        os.remove(save_folder+'\\'+i)

if "loaded_data" not in st.session_state:
     st.session_state.loaded_data = None  

st.subheader('📑Minmalistic Article/Paper RAG')
filepdf =st.sidebar.file_uploader("Upload your pdf files here" , type='pdf')
DSR1 = st.sidebar.toggle("Activate DSR1")
QMath = st.sidebar.toggle("Activate Qmath")

def main():
 with st.sidebar:
    if DSR1 and not QMath:
                    llm = OllamaLLM(model='deepseek-r1:1.5b', temperature=0)
                    st.info('model: Deepseekr1:1.5b' , icon='⚙️')
    elif QMath and not DSR1:
                  llm = OllamaLLM(model='qwen2-math:1.5b', temperature=0)
                  st.info('model: qwen2-math:1.5b' , icon='⚙️')        
    else : 
                   llm = OllamaLLM(model='qwen2.5:1.5b', temperature=0)
                   st.info('model : qwen2.5:1.5b' , icon='⚙️')
    if filepdf is not None:
       st.session_state.loaded_data = filepdf
       pdffile =load_doc(filepdf)
       st.session_state.pdf_id = pdffile
       embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v1")
       content = ""
       for page in range(len(pdffile.pages)):
                content += pdffile.pages[page].extract_text()
       text_splitter = SemanticChunker(embeddings=embeddings , breakpoint_threshold_type='standard_deviation')
       documents=text_splitter.create_documents([content])
       st.success('Pdf uploaded') 
       vectorstore = Chroma.from_documents(documents=documents , embedding=embeddings)
       retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
       
       if st.button('PDF Image Gallery'):
                  
                  img_loader(filepdf)  
                  streamImage(save_folder)            

       if st.button('Clear Gallary'):
                   clear_gallary(save_folder) 
                   st.success('pdf image gallary cleared!')   
                   st.session_state.clicked == False            
                  
       system_prompt = """
                        You a Retrieval-Augmented Generation AI , who when given pdf can retrieve the exact information needed from the pdf.
                        Use only the context that is pdf for your answers, do not make up information.

                        {context} 
                        """
       prompt = ChatPromptTemplate.from_messages([("system",system_prompt) ,] ,template_format="f-string")       
       rag_chain =  ({"context" : retriever}|prompt|llm|StrOutputParser()) 

 if "messages" not in st.session_state:
     st.session_state.messages = []

 if 'clicked' not in st.session_state:
    st.session_state.clicked = False   
                    

 user_query = st.chat_input('What are you looking for in the paper ? ')


 for message in st.session_state.messages:
                  if isinstance(message , HumanMessage):
                      with st.chat_message("Human"):
                         st.markdown(message.content)
                  elif isinstance(message , AIMessage):
                       with st.chat_message("AI"):
                           st.markdown(message.content)     

 if user_query is not None and user_query != "":
                with st.chat_message("Human"):
                    st.markdown(user_query)
                    st.session_state.messages.append(HumanMessage(user_query))  
                
                with st.chat_message("AI"):
                  response=rag_chain.invoke(user_query)
                  st.markdown(response) 
                  st.session_state.messages.append(AIMessage(response))


if  __name__== "__main__":
    if runtime.exists():
        main()
    else :    
        stcli.main_run(["miniRAG.py" , '--global.developmentMode=false'])
        sys.exit(stcli.main())
    
