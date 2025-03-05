## MiniRAG Version 1

### To open the interface just type :  <directory> / python MiniRAG.py 


This application is a pretty minimalist RAG application , that works on simple pdfs .

#### Pre-requisites:
+ Needs Ollama , download following models 
      
      +  qwen2.5:1.5b
  
      +  qwen2-math:1.5b
  
      +  deepseek-r1:1.5b
      

#### Uses : 
+ sementic chunking 
+ image extraction from simple pdf (also available in 'uploaded references' folder created)
+ image gallary deletion (deletes all the images extracted from created folder)
+ Huggingface embedding
+ Chromadb vectorspace
+ toggling between qwen2.5:1.5b , qwen2-math:1.5b , deepseek-r1:1.5b  models from Ollama (default = qwen2.5:1.5b)
  

#### Bugs :
+ Chromadb tnent error when changing pdf (needs the app needs to be exited and then try again to load chroma vectorspace again)
+ Image extraction is accurate for research papers and articles but not much for other kinds of pdf

#### Screenshots :

![Screenshot 2025-03-05 091518](https://github.com/user-attachments/assets/43aa76f3-45f3-4cd2-b205-c9288ef43b73)

![Screenshot 2025-03-05 093959](https://github.com/user-attachments/assets/773eeefd-50e9-4052-810a-e7e3f336af39)

![Screenshot 2025-03-05 094157](https://github.com/user-attachments/assets/74dae98d-e15a-4486-88de-4947b89cc387)

![Screenshot 2025-03-05 094416](https://github.com/user-attachments/assets/f1be46b1-a6f8-47e5-876f-e66485943b55)



