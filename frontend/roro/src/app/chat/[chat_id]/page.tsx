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
  
  // If params.chat_id is an array, take the first element, otherwise use it directly
  const initialChatId = (Array.isArray(params.chat_id) ? params.chat_id[0] : params.chat_id) || null;
  const [chatId, setChatId] = useState<string | null>(initialChatId);
  const [input, setInput] = useState("");
  const [model, setModel] = useState("deepseek");
  const pendingMessageRef = useRef<string>("");
  // Controls whether messages are polled
  const [pollingEnabled, setPollingEnabled] = useState(true);

  // Query current chat, enabled only when chatId exists
  const { data: chat } = useQuery({
    queryKey: ["chat", chatId],
    queryFn: () => axios.post("/api/get-chat", { chat_id: chatId }),
    enabled: !!chatId,
  });

  // Queries the chat history. refetchInterval is controlled by pollingEnabled.
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

  // Listen for preMessages to change and stop polling if the last message is assistant.
  useEffect(() => {
    if (preMessages && preMessages.data && preMessages.data.length > 0) {
      const lastMsg = preMessages.data[preMessages.data.length - 1];
      if (lastMsg.role === "assistant") {
        setPollingEnabled(false);
      }
    }
  }, [preMessages]);

  // Creating a Chat Mutation, consistent with the Home component
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
      // Polling before sending a new message
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

  // Send message Mutation, consistent with Home component
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
      // Starts polling after sending a message, waiting for a new reply
      setPollingEnabled(true);
      queryClient.invalidateQueries({ queryKey: ["messages", chatId] });
    },
  });

  // Handling message sending logic
  const handleSubmit = () => {
    if (input.trim() === "") return;
    // Starts polling before each message is sent
    setPollingEnabled(true);
    if (!chatId) {
      pendingMessageRef.current = input;
      createChat();
    } else {
      sendChat({ chatId: Number(chatId), content: input, role: "user" });
    }
    setInput("");
  };

  // Chat message list scrolls to the bottom
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

      {/* message input area */}
      <div className="bg-white flex flex-col items-center mt-4 shadow-lg border-[1px] border-gray-300 h-32 rounded-lg w-7/8">
        <textarea 
          className="bg-white w-full rounded-lg p-3 h-30 focus:outline-none"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        ></textarea>
        <div className="flex flex-row items-center justify-between w-full h-12 mb-2 bg-white">
          <div></div>
          <div 
            role="button"
            aria-label="send"
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
