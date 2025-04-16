'use client'

import React, { useState, useRef, useEffect } from "react";
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import axios from "axios";
import { useMutation, useQueryClient, useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";

export default function Home() {
  const [input, setInput] = useState("");
  const [model, setModel] = useState("deepseek");
  const queryClient = useQueryClient();
  const router = useRouter();
  const [chatId, setChatId] = useState<string | null>(null);
  const pendingMessageRef = useRef<string>("");

  // New status: Control polling enabled or not
  const [pollingEnabled, setPollingEnabled] = useState(true);

  // Create Chat Mutation: only creates chat logs, does not save messages
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
      // Starts polling before each new message is sent
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

  // Send message Mutation: unified call to /api/chat, backend handles user message saving and AI replies
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
      // Restart polling after sending a message and wait for a new reply
      setPollingEnabled(true);
      queryClient.invalidateQueries({ queryKey: ["messages", chatId] });
    },
  });

  // New query: Get chat history messages (polling control)
  const { data: messagesData } = useQuery({
    queryKey: ["messages", chatId],
    queryFn: () =>
      axios.post("/api/get-messages", { chat_id: chatId }),
    enabled: !!chatId,
    refetchInterval: pollingEnabled ? 3000 : false,
  });

  // Listen for changes to the messagesData and stop polling if the latest message is assistant.
  useEffect(() => {
    if (messagesData && messagesData.data && messagesData.data.length > 0) {
      const lastMsg = messagesData.data[messagesData.data.length - 1];
      if (lastMsg.role === "assistant") {
        setPollingEnabled(false);
      }
    }
  }, [messagesData]);

  // Handling ‘Send’ button clicks: calling /api/chat to send user messages
  const handleSubmit = () => {
    if (input.trim() === "") return;
    // 每次发送新消息前启动轮询
    setPollingEnabled(true);
    if (!chatId) {
      pendingMessageRef.current = input;
      createChat();
    } else {
      sendChat({ chatId: Number(chatId), content: input, role: "user" });
    }
    setInput("");
  };

  // Dropdown Menu Section: Using Dropdown Components to Handle Option Buttons
  const options = [
    { 
      icon: "🎓", 
      text: "Studying in Australia", 
      detail: [
        "What are the benefits of studying in Australia?",
        "Which Australian city is best for international students?",
        "What are the advantages of studying in a regional area of Australia?",
        "How does the Australian education system work?",
        "How do I choose the right university and course for me?",
        "What are the most popular courses for international students in Australia?",
        "What academic qualifications do I need to study in Australia?",
        "What scholarships are available for international students in Australia?",
        "How is the academic environment different in Australia?",
        "What should I know about my university's facilities and resources?",
        "What are the key academic dates I need to remember?",
      ]
    },
    { 
      icon: "✈️", 
      text: "Preparing for Your Journey", 
      detail: [
        "What are the different visa options for studying in Australia?",
        "How do I apply for an Australian student visa?",
        "What should I prepare before moving to Australia?",
        "What essential items should I pack for Australia?",
        "What documents do I need before traveling to Australia?",
        "How can I prepare for my first few days in Australia?",
        "What are my housing options in Australia?",
        "What are some common Aussie slang and phrases?",
        "What are the major public holidays in Australia?",
        "What are the must-have apps for living in Australia?",
      ]
    },
    { 
      icon: "🏡", 
      text: "Settling in Australia", 
      detail: [
        "What should I do in my first week after arriving in Australia?",
        "Where can I find support services for international students?",
        "How do I open a student bank account in Australia?",
        "Which mobile network and internet provider should I choose?",
        "What are some fun things to do in my city?",
        "What should I know about Australian customs and etiquette?",
        "Where can I buy affordable groceries and daily essentials?",
        "How do I use public transport in Australia?",
        "Where can I find affordable and diverse food options?",
        "What safety tips should I know for living in Australia?",
      ]
    },
    { 
      icon: "💼", 
      text: "Working & Career Development", 
      detail: [
        "How can I find part-time jobs as an international student?",
        "Can international students start a business in Australia?",
        "How can I find internships to gain experience in Australia?",
        "How do I find full-time jobs after graduation?",
        "How can I apply for a Temporary Graduate Visa?",
        "How can the Professional Year Program help my career?",
      ]
    },
    { 
      icon: "🌏", 
      text: "Life in Australia", 
      detail: [
        "What are the key facts about Australia that I should know? ",
        "How does life in Australia differ from my home country?",
        "What are some important things to know about Australia's culture, climate, and lifestyle?",
        "What are the costs of studying and living in Australia?",
        "How can I explore Australia on a budget?",
        "What are some major events and festivals happening in Australia?",
        "What are some must-visit places in Australia?",
        "How does the Australian healthcare system work for students?",
      ]
    },
    { 
      icon: "🚀", 
      text: "After Graduation & Future Pathways", 
      detail: [
        "What should I prepare for my graduation ceremony?",
        "What are my options for further study after graduation?",
        "What are my work visa options after graduation?",
        "What are the best travel destinations before leaving Australia?",
        "What are the pathways to permanent residency in Australia?",
        "What should I prepare before returning to my home country? ",
      ]
    },
  ];

  // Processing option actions
  const handleOptionAction = (selectedText: string) => {
    // Start polling for new replies
    setPollingEnabled(true);
    if (!chatId) {
      pendingMessageRef.current = selectedText;
      createChat();
    } else {
      sendChat({ chatId: Number(chatId), content: selectedText, role: "user" });
    }
  };

  return (
    <div className="h-screen w-4/5 flex flex-col items-center border border-gray-200 rounded-xl m-6">
      <div className="w-full mb-4 text-left px-5 py-4">
        <p className="text-xl">Chatting with</p>
        <h2 className="text-2xl font-semibold">roro</h2>
      </div>
      <div className="h-1/20"></div>
      <div className="w-3/4">
        <p className="text-bold text-3xl text-center mb-6">
          What can I help you with?
        </p>
        <div className="grid grid-cols-2 gap-4 w-auto">
          {options.map((option) => (
            <Dropdown
              key={option.text}
              icon={option.icon}
              text={option.text}
              detail={option.detail}
              onAction={handleOptionAction}
            />
          ))}
        </div>
        {/* Display the list of messages (if there is a chatId) */}
        {chatId && messagesData && messagesData.data && (
          <div className="mt-4">
            {messagesData.data.map((msg: any) => (
              <div
                key={msg.id || msg.content}
                className={`rounded-lg flex flex-row gap-2 ${msg.role === "assistant" ? "justify-start" : "justify-end"}`}
              >
                {msg.role === "assistant" && (
                  <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center text-white font-bold">
                    R
                  </div>
                )}
                <p className={`inline-block p-2 rounded-lg ${msg.role === "assistant" ? "bg-white" : "bg-[#7FCD89]"}`}>
                  {msg.content}
                </p>
                {msg.role !== "assistant" && (
                  <div className="flex-shrink-0 h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold">
                    U
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        <div className="bg-white flex flex-col items-center mt-4 shadow-lg border-[1px] border-gray-300 h-32 rounded-lg">
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
    </div>
  );
}

/** Dropdown component: display dropdown menu on mouse hover, support detail as array */
function Dropdown({ icon, text, detail, onAction }: { 
  icon: string; 
  text: string; 
  detail: string | string[]; 
  onAction: (selectedText: string) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <button
        className="flex flex-row text-sm items-center gap-0.5 
          bg-white shadow-md px-1 py-3 rounded-lg w-full h-auto 
          text-gray-700 font-medium hover:bg-gray-100 transition 
          break-words"
      >
        <span className="text-xl">{icon}</span>
        <span className="ml-1 text-left flex-wrap whitespace-normal leading-none">
          {text}
        </span>
      </button>
      {isOpen && (
        <div
          className="absolute top-full left-0 w-auto bg-white border border-gray-300 rounded-lg shadow-lg p-3 z-40"
          style={{ minWidth: '200px' }}
        >
          {Array.isArray(detail) ? (
            <ul className="space-y-2">
              {detail.map((item, index) => (
                <li key={index}>
                  <button
                    className="w-full text-left px-3 py-1 text-sm hover:bg-gray-100 cursor-pointer whitespace-nowrap"
                    onClick={() => onAction(item)}
                  >
                    {item}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-700 whitespace-pre-wrap">
              {detail}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
// export { Dropdown };
