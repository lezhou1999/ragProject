import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi 


INDEX_PATH = "C:/coding/yoSmartRag/yosmart_index.index"
CHUNKS_PATH = "C:/coding/yoSmartRag/chunks.pkl"
#faiss 不支持非ascii字符文件，所以index文件的路径中不可以有中文

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")





TOP_K = 10 #返回前k个最相似的chunk

#print("Current Working Directory:", os.getcwd())
#print("Index file exists:", os.path.exists(INDEX_PATH))

#=== openai 客户端===
# openai的 apikey可以存入全局变量
client = OpenAI(api_key=api_key)

#==  upload embedding model ===
#bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 用于embedding
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # 用于rerank（可选）

# === 加载 index 和 chunks ===
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

chunk_texts = [c if isinstance(c, str) else c.get("content", "") for c in chunks]
tokenized_corpus = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)   # <=== 这里初始化一次，全局可用

def get_openai_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 或 "text-embedding-3-large"
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


def retrieve_chunks(query: str, top_k: int = TOP_K):
    # Step 1: 向量化查询
    query_embedding = get_openai_embedding(query).reshape(1, -1)

    # Step 2: FAISS 召回
    D, I = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Step 3: CrossEncoder 评分
    reranked = []
    for chunk in retrieved_chunks:
        score = float(cross_encoder.predict([[query, chunk]]))
        reranked.append((chunk, score))

    # Step 4: BM25 fallback
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 获取 BM25 top-k
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_results = [(chunks[i], bm25_scores[i]) for i in bm25_top_idx]

    # Step 5: Hybrid 融合
    hybrid = []

    # 先加 embedding + CrossEncoder 的结果
    for chunk, score in reranked:
        hybrid.append((chunk, score))

    # 再加 BM25 fallback（如果重复 chunk，用更高分数）
    for chunk, score in bm25_results:
        found = False
        for i, (c, s) in enumerate(hybrid):
            if c == chunk:
                hybrid[i] = (c, max(s, score + 5))  # 提升 BM25 权重
                found = True
                break
        if not found:
            hybrid.append((chunk, score + 5))  # 新增 BM25 结果

    # Step 6: 排序，返回 top_k
    hybrid = sorted(hybrid, key=lambda x: x[1], reverse=True)[:top_k]

    return hybrid


if __name__ == "__main__":
    #for test
    query = "how to use CSDevice "
    results = retrieve_chunks(query)

    print("Top chunks:\n")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"Rank {i} | Score: {score:.4f}")
        print(chunk)
        print("=" * 80)