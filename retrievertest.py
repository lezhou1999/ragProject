import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi   # ✅ BM25

INDEX_PATH = "C:/coding/yoSmartRag/yosmart_index.index"
CHUNKS_PATH = "C:/coding/yoSmartRag/chunks.pkl"

TOP_K = 3  # 返回前k个最相似的chunk

# == 载入模型 ==
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")  
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# === 加载 index 和 chunks ===
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# ✅ 初始化 BM25 全局对象
# 假设 chunks 是字符串或 dict
chunk_texts = [c if isinstance(c, str) else c.get("content", "") for c in chunks]
tokenized_corpus = [text.lower().split() for text in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus)   # <=== 这里初始化一次，全局可用

def retrieve_chunks(query: str, top_k: int = TOP_K):
    """
    执行混合 RAG 检索: FAISS + BM25 fallback
    """
    # Step 1: 向量检索
    query_embedding = bi_encoder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    D, I = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Step 2: 如果 FAISS 结果为空，使用 BM25 fallback
    if not retrieved_chunks or all(len(str(c)) < 10 for c in retrieved_chunks):
        print("⚠️ FAISS 没找到合适结果，使用 BM25 fallback...")
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(bm25_scores)[::-1][:top_k]
        retrieved_chunks = [chunks[i] for i in top_n]

    # Step 3: CrossEncoder 重排
    pairs = [[query, str(chunk)] for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)

    return reranked

if __name__ == "__main__":
    query = "CSDevice.sendCommand"
    results = retrieve_chunks(query)

    print("Top chunks:\n")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"Rank {i} | Score: {score:.4f}")
        print(chunk)
        print("=" * 80)
