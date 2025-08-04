use：
 .\.venv\Scripts\Activate.ps1 作为虚拟环境的入口\\


 需要设置.env 文件，使用自己的openai api key


 fast api 的启动：
 python -m uvicorn rag_fastApi:app --host 0.0.0.0 --port 8000 --reload

 前端服务器：
 rag-chat 
npm run dev
