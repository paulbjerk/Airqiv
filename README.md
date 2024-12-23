# Airqiv AI-Assistant
The Airqiv AI-Assistant Document Explorer is a Retrieval Augmented Generation (RAG) process for Apple M-series Mac computers. It uses Ollama and ChromaDB, and is written in Python. It works offline with local files. An internet connection is only needed for setup.
To use you can dowload the four python files, and the CSV file as a model for ingest. This CSV file contains the text extracted via an OCR process from photos of documents at the US National Archive (NARA II) in College Park, MD. Photos of the archival box and first full document (five photos which are the first five entries in the spreadsheet) are also included here.

To use: This only works on an M-series Apple Macintosh computer (2020 or later). 

1. Create a folder called airqiv in your main user folder.
2. Then download and move these four files into it: 1) setup.py 2) ingest.py 3) asst.py 4) delete_collection.py
3. Put the nara_tanzania_RG59-67-69-Box2513-contd-2.csv (the other CSV creates an error) in the ai-assistant folder in your main user folder. You can also try using this Apple Short cut to extract text from interview transcripts of Oral Histories of Vietnam War veterans collected by the The Oral History Project of the Vietnam Center and Sam Johnson Vietnam Archive at Texas Tech University: https://www.icloud.com/shortcuts/07c846c77c0546199a2e432c83997645 This shortcut will cycle through 50 transcripts, starting from the the transcript number entered in the first step.
4. Download Ollama from ollama.com and put it in your Applications folder (you’ll need an admin password for this...I’m not sure that’s altogether necessary as the setup file also installs Ollama for the local user. Either way, you need Ollama to run this)
5. Open the Terminal application (from the Utilities folder in your Applications folder)
6. Type - cd airqiv
7. Type - python3 setup.py
8. Run the set up program. You need to be online for this as it will download chromadb, ollama, and some language models from ollama.
9. When prompted enter your M-series chip and RAM memory (available from "About this Mac" under the Apple menu)
10. In terminal type - python3 ingest.py
11. Ingest the CSV file in the folder by copying and pasting just the file name (i.e. enter only nara_tanzania_RG59-67-69-Box2513 ...without the .csv suffix)
12. When prompted to enter archive name and topical collection, type (lowercase, just the word, not the dashes): - nara - for the archive and - tanzania - for the topical collection (it also asks for sub-collection, for this type snf... i.e. the State Department Subject Numeric File) 
13. In Terminal type - python3 asst.py
14. Explore the documents by typing in the lowercase archive abbreviation -  nara
15. Follow the prompts in the asst app. They mostly work, but there are bound to be errors. Let me know if you encounter errors

