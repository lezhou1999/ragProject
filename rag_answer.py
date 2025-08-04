from retriever import retrieve_chunks
from transformers import pipeline
from openai import OpenAI
from dotenv import load_dotenv
import os
# How to get the firmware version of a dimmer?
# 初始化本地或 OpenAI QA 模型（根据你想要的方式）
#qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

load_dotenv()

# 获取 API key

#print("API Key loaded:", os.getenv("OPENAI_API_KEY"))
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


def answer_query(query: str):
    '''
    chunks = retrieve_chunks(query, top_k=3)
    context = "\n\n".join([chunk[0] for chunk in chunks])
   
    result = qa_pipeline({
        "question": query,
        "context": context
    })
    return result["answer"]
    '''
    results = retrieve_chunks(query)
    chunks = [chunk for chunk, _ in results]   # 取出 chunk（丢掉 score）


    # Step 2: 拼接上下文
    # 如果 chunks 是 dict，就取 content；如果是 str，直接用
    context = "\n\n".join([c["content"] if isinstance(c, dict) else str(c) for c in chunks])


    # Step 3: 构造 Prompt
    prompt = f"""
You are an assistant with access to YoSmart device API documentation.
Use the following context to answer the question as accurately as possible.
If the answer is not in the context, say "I don't know from docs."


Question: {query}


Context:
{context}


Answer:
"""


    # Step 4: 调用 OpenAI GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # ✅ 成本低，也足够强
        messages=[
            {"role": "system", "content": "You are a helpful assistant for YoSmart API documentation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )


    return response.choices[0].message.content.strip()




if __name__ == "__main__":
    query = input(" Your Question: ")
    print("🧾 Answer:", answer_query(query))
