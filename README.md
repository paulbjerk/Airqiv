# AI-Assistant
This is a Retrieval Augmented Generation (RAG) process for Apple M-series Mac computers. It uses Ollama and ChromaDB, and is written in Python.
To use you can dowload the setup.py, ingest.py, and asst.py files, and the NARA-RG59-67-69-Box2513.csv file as a model for ingest. This CSV file contains the text extracted via an OCR process from photos of documents at the US National Archive (NARA II) in College Park, MD. 

To use: This only works on an M-series Apple Macintosh computer (2020 or later). 

1. Create a folder called ai-assistant in your main user folder and put these four files in it. 
2. Download Ollama from ollama.com and put it in your Applications folder (you’ll need an admin password for this…I’m not sure that’s altogether necessary as the setup file also installs Ollama I think just in the user’s applications…but either way, you need Ollama to run this)
3. Open the Terminal application (from the Utilities folder in your Applications folder)
4. Type - cd ai-assistant
5. Type - python3 setup.py
6. Run the set up and it will tell you about the next step
7. In terminal type - python3 ingest.py when prompted about archive and topical collection type - nara - for the archive and - tanzania - for the topical collection
8. Ingest the CSV file in the folder by copying and pasting just the file name (without the .csv suffix)
9 then in Terminal type - python3 asst.py and explore the documents by typing in - all-nara-documents
10. Follow the prompts in the app… they mostly work, but there are bound to be errors. Let me know if you encounter errors

