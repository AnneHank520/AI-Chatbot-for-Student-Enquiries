'use client';

import { useChat } from '@ai-sdk/react';
import { useEffect, useRef, useState } from 'react';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import { useParams } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

export default function Page() {
  const { chat_id } = useParams();

  // 用于追踪最后处理的用户消息 id，防止重复触发
  const lastProcessedUserMessageId = useRef<string | null>(null);

  // **聊天容器**的 ref，用于手动设置 scrollTop
  const chatContainerRef = useRef<HTMLDivElement | null>(null);

  // 获取聊天信息
  const { data: chat } = useQuery({
    queryKey: ['chat', chat_id],
    queryFn: () => axios.post(`/api/get-chat`, { chat_id }),
  });

  // 获取聊天历史消息
  const { data: preMessages } = useQuery({
    queryKey: ['messages', chat_id],
    queryFn: () =>
      axios.post(`/api/get-messages`, {
        chat_id,
        chat_user_id: chat?.data?.userId,
      }),
    enabled: !!chat?.data?.id,
    refetchInterval: 1000, // 每1秒自动刷新一次
  });

  const [model, setModel] = useState("deepseek");

  // 判断是否加载完成
  const isLoading = !preMessages?.data;

  // 始终调用 useChat，并传入初始消息（空数组或加载到的消息）
  const { messages, input, handleInputChange, handleSubmit, append } = useChat({
    body: { model, chat_id, chat_user_id: chat?.data?.userId },
    initialMessages: preMessages?.data || [],
  });

  // 自动回复逻辑：当最后一条消息为用户消息且没有 AI 回复时，触发自动回复
  useEffect(() => {
    if (isLoading) return;

    const autoReply = async () => {
      try {
        const res = await axios.post("/api/chat", {
          chat_id,
          model,
          messages,
          chat_user_id: chat?.data?.userId,
        });
        if (res.data?.content) {
          await append({ role: "assistant", content: res.data.content });
        }
      } catch (err) {
        console.error("Auto reply error:", err);
      }
    };

    if (messages && messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg.role === "user") {
        // 若对话中至少有两条消息，并且倒数第二条是 assistant，则说明这条用户消息已回复
        if (messages.length > 1) {
          const secondLast = messages[messages.length - 2];
          if (secondLast.role === "assistant") return;
        }
        // 否则如果这条用户消息还未被处理，则触发自动回复
        if (lastProcessedUserMessageId.current !== lastMsg.id) {
          lastProcessedUserMessageId.current = lastMsg.id;
          autoReply();
        }
      }
    }
  }, [isLoading, messages, chat_id, model, chat, append]);

  // **在每次 messages 更新后，让聊天容器滚动到底部**（只滚内部容器）
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <div ref={chatContainerRef} className="h-screen w-4/5 flex flex-col items-center border border-gray-200 rounded-xl m-6 overflow-y-auto">
      <div className="w-full mb-4 text-left px-5 py-4">
        <p className="text-xl">Chatting with</p>
        <h2 className="text-2xl font-semibold">roro</h2>
      </div>

      <div className="flex flex-col w-7/8 gap-8 justify-between flex-1">
        <div className="flex flex-col gap-8 flex-1">
          {messages.map((message) => (
            <div
              key={message.id || message.content}
              className={`rounded-lg flex flex-row gap-2 ${
                message.role === "assistant" ? "justify-start" : "justify-end"
              }`}
            >
              {message.role === "assistant" && (
                <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center text-white font-bold">
                  R
                </div>
              )}
              <p
                className={`inline-block p-2 rounded-lg ${
                  message.role === "assistant" ? "bg-white" : "bg-[#7FCD89]"
                }`}
              >
                {message.content}
              </p>
              {message.role !== "assistant" && (
                <div className="flex-shrink-0 h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold">
                  U
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="h-4"></div>

      {/* 输入区域 */}
      <div className="bg-white flex flex-col items-center mt-4 shadow-lg border-[1px] border-gray-300 h-32 rounded-lg w-7/8">
        <textarea
          className="bg-white w-full rounded-lg p-3 h-30 focus:outline-none"
          value={input}
          onChange={handleInputChange}
        />
        <div className="flex flex-row items-center justify-between w-full h-12 mb-2 bg-white">
          <div></div>
          <div
            className="flex items-center justify-center mr-4 p-1 rounded-full bg-[#244E48] cursor-pointer"
            onClick={handleSubmit}
          >
            <ArrowUpwardIcon className="text-white" />
          </div>
        </div>
      </div>
    </div>
  );
}
