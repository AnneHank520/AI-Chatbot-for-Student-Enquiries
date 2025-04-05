'use client';

import React, { useState, useRef, useEffect } from "react";
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import axios from "axios";
import { useMutation, useQueryClient, useQuery } from "@tanstack/react-query";
import { useParams, useRouter } from "next/navigation";

export default function Page() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  
  // 如果 params.chat_id 是数组，则取第一个元素，否则直接使用
  const initialChatId = (Array.isArray(params.chat_id) ? params.chat_id[0] : params.chat_id) || null;
  const [chatId, setChatId] = useState<string | null>(initialChatId);
  const [input, setInput] = useState("");
  const [model, setModel] = useState("deepseek");
  const pendingMessageRef = useRef<string>("");
  // 控制是否轮询消息
  const [pollingEnabled, setPollingEnabled] = useState(true);

  // 查询当前聊天信息，仅在 chatId 存在时启用
  const { data: chat } = useQuery({
    queryKey: ["chat", chatId],
    queryFn: () => axios.post("/api/get-chat", { chat_id: chatId }),
    enabled: !!chatId,
  });

  // 查询聊天历史消息，refetchInterval 根据 pollingEnabled 控制
  const { data: preMessages } = useQuery({
    queryKey: ["messages", chatId],
    queryFn: () =>
      axios.post("/api/get-messages", {
        chat_id: chatId,
        chat_user_id: chat?.data?.userId,
      }),
    enabled: !!chat?.data?.id,
    refetchInterval: pollingEnabled ? 3000 : false,
  });

  // 监听 preMessages 变化，若最后一条消息为 assistant 则停止轮询
  useEffect(() => {
    if (preMessages && preMessages.data && preMessages.data.length > 0) {
      const lastMsg = preMessages.data[preMessages.data.length - 1];
      if (lastMsg.role === "assistant") {
        setPollingEnabled(false);
      }
    }
  }, [preMessages]);

  // 创建聊天 Mutation，与 Home 组件一致
  const { mutate: createChat } = useMutation({
    mutationFn: async () => {
      return axios.post("/api/create-chat", {
        title: "Chat",
        model: model,
      });
    },
    onSuccess: (res) => {
      const newChatId = res.data.id;
      setChatId(newChatId);
      // 发送新消息前启动轮询
      setPollingEnabled(true);
      if (pendingMessageRef.current) {
        sendChat({
          chatId: Number(newChatId),
          content: pendingMessageRef.current,
          role: "user",
        });
        pendingMessageRef.current = "";
      }
      router.push(`/chat/${newChatId}`);
      queryClient.invalidateQueries({ queryKey: ["chats"] });
    },
  });

  // 发送消息 Mutation，与 Home 组件一致
  const { mutate: sendChat } = useMutation({
    mutationFn: async (vars: { chatId: number; content: string; role: string }) => {
      const payload = {
        chat_id: vars.chatId,
        chat_user_id: undefined,
        model: model,
        messages: [{ role: vars.role, content: vars.content }],
      };
      return axios.post("/api/chat", payload);
    },
    onSuccess: () => {
      // 发送消息后启动轮询，等待新的回复
      setPollingEnabled(true);
      queryClient.invalidateQueries({ queryKey: ["messages", chatId] });
    },
  });

  // 处理消息发送逻辑
  const handleSubmit = () => {
    if (input.trim() === "") return;
    // 每次发送消息前启动轮询
    setPollingEnabled(true);
    if (!chatId) {
      pendingMessageRef.current = input;
      createChat();
    } else {
      sendChat({ chatId: Number(chatId), content: input, role: "user" });
    }
    setInput("");
  };

  // 聊天消息列表滚动到底部
  const chatContainerRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [preMessages]);

  if (!preMessages?.data) {
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
          {preMessages.data.map((message: any) => (
            <div
              key={message.id || message.content}
              className={`rounded-lg flex flex-row gap-2 ${message.role === "assistant" ? "justify-start" : "justify-end"}`}
            >
              {message.role === "assistant" && (
                <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center text-white font-bold">
                  R
                </div>
              )}
              <div className={`inline-block p-2 rounded-lg ${message.role === "assistant" ? "bg-white" : "bg-[#7FCD89]"}`}>
                {message.role === "assistant" ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  <span>{message.content}</span>
                )}
              </div>
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

      {/* 消息输入区域 */}
      <div className="bg-white flex flex-col items-center mt-4 shadow-lg border-[1px] border-gray-300 h-32 rounded-lg w-7/8">
        <textarea 
          className="bg-white w-full rounded-lg p-3 h-30 focus:outline-none"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        ></textarea>
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
