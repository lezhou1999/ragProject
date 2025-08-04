import os
import re
import faiss
from openai import OpenAI
import glob
import numpy as np
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
import pickle

load_dotenv()

# 2. 获取 API key
api_key = os.getenv("OPENAI_API_KEY")

# 3. 初始化 OpenAI client
client = OpenAI(api_key=api_key)


EMBEDDING_MODEL = "text-embedding-ada-002"


#EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")




def load_chunks_from_files(directory):
    filePrefix = "data"
    fileSuffix = ".txt"
    fileCount = 21
    chunks = []
    for i in range(1,fileCount+1):
        filepath = os.path.join(directory, f"{filePrefix}{i}{fileSuffix}")
        with open(filepath,"r",encoding="utf-8") as f:
            content = f.read()
            # 使用正则提取每个 CHUNK 内容
            split_chunks = re.findall(r"=== CHUNK:[^\n]*\n(.*?)(?=\n=== CHUNK:|\Z)", content, re.DOTALL)
            cleaned_chunks = [chunk.strip() for chunk in split_chunks if chunk.strip()]
            chunks.extend(cleaned_chunks)
    return chunks




def get_embedding(text:str):
    return client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    ).data[0].embedding


'''




def get_embedding(text: str) -> List[float]:
    return EMBEDDING_MODEL.encode(text).tolist()


'''


def build_faiss_index(chunks: List[str], index_path: str = "yosmart_index.index"):
    print(" Generating embeddings...")
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])


    print(f" Embedding dimension: {dimension}, total chunks: {len(embeddings)}")
   
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))


    faiss.write_index(index, index_path)
    print(f" FAISS index saved to: {index_path}")




if __name__ == "__main__":
    folder = r"C:\coding\yoSmartRag\data\yosmart-docs"  # 替换为实际文件夹路径
    chunks = load_chunks_from_files(folder)
    build_faiss_index(chunks)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
