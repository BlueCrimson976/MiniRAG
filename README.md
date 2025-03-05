## MiniRAG Version 1


This application is a pretty minimalist RAG application , that works on simple pdfs .

#### Pre-requisites:
+ Needs Ollama , download following models 
      + qwen2.5:1.5b 
      + qwen2-math:1.5b
      + deepseek-r1:1.5b

#### Uses : 
+ sementic chunking 
+ image extraction from simple pdf (also available in 'uploaded references' folder created)
+ image gallary deletion (deletes all the images extracted from created folder)
+ Chromadb vectorspace
+ toggling between qwen2.5:1.5b , qwen2-math:1.5b , deepseek-r1:1.5b  models from Ollama (default = qwen2.5:1.5b)
  

#### Bugs :
+ Chromadb tnent error when changing pdf (needs the app needs to be exited and then try again to load chroma vectorspace again)
+ Image extraction is accurate for research papers and articles but not much for other kinds of pdf

#### Screenshots :

![Screenshot 2025-03-05 091518](https://github.com/user-attachments/assets/43aa76f3-45f3-4cd2-b205-c9288ef43b73)

