import pandas as pd
import os
import argparse
from tqdm import tqdm
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


df = pd.read_csv("./../QA_chatbot_dataset_Akshaya.csv")
df_ans = df['QuestionAnswer']
# df_qn = df['Question']
knowledge = "\n".join(df_ans)
# question = "\n".join(df_qn)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 0,
    length_function = len,
)
knowledge = text_splitter.split_text(knowledge)

documents_directory = "documents"
collection_name = "akshaya_bot"
persist_directory = "."

documents = []
# metadatas = []
# files = os.listdir(documents_directory)

# for filename in files:
#     with open(f"{documents_directory}/{filename}", "r", encoding="utf-8") as file:
#         for line_number, line in enumerate(
#             tqdm(file.readlines(), desc=f"Reading {filename}"), 1
#         ):
#             line = line.strip()
#             if len(line) == 0:
#                 continue
#             documents.append(line)
#             metadatas.append({"filename": filename, "line_number": line_number})

documents.append(knowledge)
print(len(documents[0]))
# print(documents)
# metadatas.append(question)

client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection(name=collection_name)


count = collection.count()
print(f"Collection already contains {count} documents")
ids = [str(i) for i in range(count, count + len(documents[0]))]

for i in tqdm(
    range(0, len(documents), 50), desc="Adding documents", unit_scale=100
):
    collection.add(
        ids=ids[i : i + 50],
        documents=documents[0][i : i + 50],
        # metadatas=metadatas[i : i + 50],  # type: ignore
    )
new_count = collection.count()
print(f"Added {new_count - count} documents")
print("sucessful")



