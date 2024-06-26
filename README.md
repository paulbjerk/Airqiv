# AI-Assistant
This is a Retrieval Augmented Generation (RAG) process for Apple M-series Mac computers. It uses Ollama and ChromaDB, and is written in Python.
To use you can dowload the three python files, and the CSV file as a model for ingest. This CSV file contains the text extracted via an OCR process from photos of documents at the US National Archive (NARA II) in College Park, MD. Photos of the archival box and first full document (five photos which are the first five entries in the spreadsheet) are also included here.

To use: This only works on an M-series Apple Macintosh computer (2020 or later). 

1. Create a folder called ai-assistant in your main user folder and then download and move these four files into it: 1) setup.py 2) ingest.py 3) asst.py 4) NARA-RG59-67-69-Box2513.csv
2. Create a folder inside the ai-assistant folder called - nara - and create another folder inside that one called - tanzania -  (i.e. user > ai-assistant > nara > tanzania )
3. Put the csv file in in the tanzania folder  ... I intend to create some functions in the setup and ingest processes that will automatically create this folder hierarchy, but have not yet done so. 
4. Download Ollama from ollama.com and put it in your Applications folder (you’ll need an admin password for this…I’m not sure that’s altogether necessary as the setup file also installs Ollama for the local user. Either way, you need Ollama to run this)
5. Open the Terminal application (from the Utilities folder in your Applications folder)
6. Type - cd ai-assistant
7. Type - python3 setup.py
8. Run the set up and it will tell you about the next step
9. In terminal type - python3 ingest.py
10. When prompted to enter archive name and topical collection, type - nara - for the archive and - tanzania - for the topical collection
11. Ingest the CSV file in the folder by copying and pasting just the file name (i.e. enter only NARA-RG59-67-69-Box2513 ...without the .csv suffix)
12. In Terminal type - python3 asst.py
13. Explore the documents by typing in -   all-nara-documents
14. Follow the prompts in the asst app. They mostly work, but there are bound to be errors. Let me know if you encounter errors

