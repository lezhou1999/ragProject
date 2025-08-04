import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import axios from "axios"

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
  if (!input.trim()) return;

  const userInput = input;  // 临时保存用户输入
  setInput("");  // 立即清空输入框

  // 前端加入用户输入
  const newMessages = [...messages, { sender: "user", text: userInput }];
  setMessages(newMessages);

  try {
    const response = await axios.post("http://localhost:9000/ask", {
      query: userInput,   // ✅ 用 userInput，而不是 input
    });

    const botReply = response.data.answer || "No answer found.";
    setMessages([...newMessages, { sender: "bot", text: botReply }]);
  } catch (error) {
    console.error(error);
    setMessages([
      ...newMessages,
      { sender: "bot", text: "Error: could not reach backend." },
    ]);
  }
  };

  return (
    <div style={{ maxWidth: "600px", margin: "20px auto", fontFamily: "Arial" }}>
      <h2>🤖 RAG Chat Assistant</h2>
      <div
        style={{
          border: "1px solid #ccc",
          padding: "10px",
          height: "400px",
          overflowY: "auto",
          marginBottom: "10px",
        }}
      >
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              textAlign: msg.sender === "user" ? "right" : "left",
              margin: "5px 0",
            }}
          >
            <b>{msg.sender === "user" ? "You" : "Bot"}:</b> {msg.text}
          </div>
        ))}
      </div>

      <div style={{ display: "flex" }}>
        <input
          style={{ flex: 1, padding: "10px" }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage} style={{ padding: "10px 20px" }}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;