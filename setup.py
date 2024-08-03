#import csv
import os

print("\n           - - The Airqiv Document Explorer  - -       ")
print(" - - Artificially Intelligent Retrieval Query Interpretive Visualizer - -")
print("                     - - airqiv.com  - -       ")
print("                        - - :-) - -         \n")
print("\nArqiv AI-Assistant Document Explorer")
print("Copyright (c) <2024>, <Paul Bjerk>")
print("All rights reserved.")
print("\nThis source code is licensed under the BSD2-style license found at https://opensource.org/license/bsd-2-clause .\n")
print("The app leverages open-sourced LLMs using the Ollama app and a vector database using ChromaDB")
print("\n The documents returned and summarized by this Document Explorer are copyright of the authors and archival custodian.\n")

embed_model = "mxbai-embed-large"
embed_model_dimensions = "1024"
embed_model_layers = "24"
inference_model = "phi3:3.8b-mini-128k-instruct-q5_K_M"
inference_model_short, inference_model_detail = inference_model.split(":")
userresponse = ""

print("\nThis setup process will take 10-15 minutes. You will need a 2020 or later Apple Mac with an M-Series chip. \nIt will install all the needed pieces (dependencies) for running the two apps.\n Before running this, you must first install Ollama from https://ollama.com \n ")
print("This will install ChromaDB first, and then install two AI language models from Ollama.\n"
      "First install Ollama and move the ai-assistant folder into your main user folder, \nthen run this set up app in the Mac Terminal by typing:\n cd ai-assistant \n python3 setup.py \n ")

userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
if userresponse == "q":
    exit()
else:
    print("Let's continue.\n")

print("Click on the Apple icon in the upper left corner, and click on -- About This Mac.")
mac_model = input ("What chip does your Mac have? Type M1 or M2, or M3: ")
ram_memory_input = input ("How many GB of RAM memory does your Mac have? (type a number): ")
ram_memory = int(ram_memory_input)

#install chromadb and the underlying chat model (LLM: phi3)
print("Make sure the above information was entered correctly. \nThis next step will run automatically and take a 5-10 minutes depending on system and internet speed")

userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
if userresponse == "q":
    exit()
else:
    print("Let's continue.\n")

#os.system("cd")
os.system("pip3 install chromadb")
print("ChromaDB has been installed. For more information see https://docs.trychroma.com \n")
os.system("pip3 install ollama")
print("Ollama has been installed. For more information see https://ollama.com \n")
os.system("ollama pull "+ embed_model)
print("The embedding model, "+embed_model+" has been installed. \nFor more information see https://ollama.com/blog/embedding-models \n")
os.system("ollama pull " + inference_model)
print("The basic LLM inference model, "+inference_model+" has been installed. \nFor more information see https://ollama.com/library/"+inference_model_short+".\n")

# functions used

def create_csv(collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","URL","COPYRIGHT","ARCHIVE","TOPIC","SUBCOLLECTION",]
    with open(str("all-"+collection+"-documents.csv"), mode="w", newline="") as new_file:
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        all_files.writeheader()

def create_folder(archive_collection):
    os.system("mkdir "+archive_collection)

def create_sub_folder(archive_collection, topic_collection):
    os.system("mkdir "+archive_collection+"/"+topic_collection)

# These create models (modelfile) with context lengths appropriate to the user's RAM memory
#https://github.com/ollama/ollama/blob/main/docs/modelfile.md

def create_small_model_template ():
    with open("model-template.txt", "w") as file:
        file.write("""FROM phi3:3.8b-mini-128k-instruct-q5_K_M
TEMPLATE "{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"
PARAMETER stop <|end|>
PARAMETER stop <|user|>
PARAMETER stop <|assistant|>
PARAMETER num_ctx 2048
PARAMETER repeat_last_n 256""")

def create_med_model_template ():
    with open("model-template.txt", "w") as file:
        file.write("""FROM phi3:3.8b-mini-128k-instruct-q5_K_M
TEMPLATE "{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"
PARAMETER stop <|end|>
PARAMETER stop <|user|>
PARAMETER stop <|assistant|>
PARAMETER num_ctx 4096
PARAMETER repeat_last_n 512""")

def create_expanded_model_template ():
    with open("model-template.txt", "w") as file:
        file.write("""FROM phi3:3.8b-mini-128k-instruct-q5_K_M
TEMPLATE "{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"
PARAMETER stop <|end|>
PARAMETER stop <|user|>
PARAMETER stop <|assistant|>
PARAMETER num_ctx 8192
PARAMETER repeat_last_n 1024""")

def create_large_model_template ():
    os.system("ollama pull phi3:14b-medium-128k-instruct-q4_K_M")
    with open("model-template.txt", "w") as file:
        file.write("""FROM phi3:14b-medium-128k-instruct-q4_K_M
TEMPLATE "{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"
PARAMETER stop <|end|>
PARAMETER stop <|user|>
PARAMETER stop <|assistant|>
PARAMETER num_ctx 12288
PARAMETER repeat_last_n 1536""")

def create_max_model_template ():
    os.system("ollama pull phi3:14b-medium-128k-instruct-q4_K_M")
    with open("model-template.txt", "w") as file:
        file.write("""FROM phi3:14b-medium-128k-instruct-q4_K_M
TEMPLATE "{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"
PARAMETER stop <|end|>
PARAMETER stop <|user|>
PARAMETER stop <|assistant|>
PARAMETER num_ctx 16384
PARAMETER repeat_last_n 2048""")

if ram_memory < 9:
    if os.path.exists("model-template.txt"):
        os.remove("model-template.txt")
        create_small_model_template()
        inference_model_window = "2k tokens"
        os.system("ollama create phi3-2k -f model-template.txt")
        os.system("ollama show --modelfile phi3-2k")
    else:
        create_small_model_template()
        inference_model_window = "2k tokens"
        os.system("ollama create phi3-2k -f model-template.txt")
        os.system("ollama show --modelfile phi3-2k")
elif 9<ram_memory<13:
    if os.path.exists("model-template.txt"):
        os.remove("model-template.txt")
        create_med_model_template()
        inference_model_window = "4k tokens"
        os.system("ollama create phi3-4k -f model-template.txt")
        os.system("ollama show --modelfile phi3-4k")
    else:
        create_med_model_template()
        inference_model_window = "4k tokens"
        os.system("ollama create phi3-4k -f model-template.txt")
        os.system("ollama show --modelfile phi3-4k")
elif 13<ram_memory<17:
    if os.path.exists("model-template.txt"):
        os.remove("model-template.txt")
        create_expanded_model_template()
        inference_model_window = "8k tokens"
        os.system("ollama create phi3-8k -f model-template.txt")
        os.system("ollama show --modelfile phi3-8k")
    else:
        create_expanded_model_template()
        inference_model_window = "8k tokens"
        os.system("ollama create phi3-8k -f model-template.txt")
        os.system("ollama show --modelfile phi3-8k")
elif 17<ram_memory<25:
    if os.path.exists("model-template.txt"):
        os.remove("model-template.txt")
        create_large_model_template()
        inference_model_window = "12k tokens"
        os.system("ollama create phi3-14b-12k -f model-template.txt")
        os.system("ollama show --modelfile phi3-14b-12k")
    else:
        create_large_model_template()
        inference_model_window = "12k tokens"
        os.system("ollama create phi3-14b-12k -f model-template.txt")
        os.system("ollama show --modelfile phi3-14b-12k")
elif ram_memory > 25:
    if os.path.exists("model-template.txt"):
        os.remove("model-template.txt")
        create_max_model_template()
        inference_model_window = "16k tokens"
        os.system("ollama create phi3-14b-16k -f model-template.txt")
        os.system("ollama show --modelfile phi3-14b-16k")
    else:
        create_max_model_template()
        inference_model_window = "16k tokens"
        os.system("ollama create phi3-14b-16k -f model-template.txt")
        os.system("ollama show --modelfile phi3-14b-16k")


print("\nThe information above summarizes the inference model we will use to explore the retrieved documents.\n The model allows a context length of " +inference_model_window+ ", which represent words or parts of words. \nDivide tokens by 1.5 to get approximate number of words that the model can analyze in the retrieved documents.\nEach page of these documents contains an average of 350 words, so a 2k model analyzes about 4 pages at a time, and an 8k model about 16 pages.\n")

print("\nNow type python3 ingest.py in order to begin ingesting CSV documents.")

#this removes a temporary file created for the purpose of creating the modelfile
os.remove("model-template.txt")
exit()
