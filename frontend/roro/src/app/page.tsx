'use client'

import { useState } from "react";
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
// import { title } from "process";
// import AddReactionIcon from '@mui/icons-material/AddReaction';
import axios from "axios";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";


export default function Home() {

  const [input, setInput] = useState("")
  const [model, setModel] = useState("deepseek")

  const queryClient = useQueryClient()
  const router = useRouter()

  // Mutation
  const {mutate: createChat} = useMutation({
    mutationFn: async() => {
      return axios.post('/api/create-chat', {
        title: input,
        model: model
      })
    },
    onSuccess: (res) => {
      router.push(`/chat/${res.data.id}`)
      queryClient.invalidateQueries({ queryKey: ['chats'] })
    },
  })

  const handleSubmit = () => {
    if (input.trim() === "") {
      return
    }

    createChat()

  }

  const options = [
    { icon: "🌿", text: "Work Life Balance" },
    { icon: "🧠", text: "Mental Wellbeing" },
    { icon: "💨", text: "Stress & Overwhelm" },
    { icon: "🌞", text: "Psychosocial Hazards" },
    { icon: "⚖️", text: "Conflict Resolution" },
    { icon: "🚩", text: "Difficult Conversation" },
    { icon: "💜", text: "Lack of Appreciation" },
    { icon: "💚", text: "Loneliness" },
  ];

  return (
    <div className="h-screen flex flex-col items-center">
      <div className="flex items-center justify-center">
          <img
          src="/images/Logo Name.svg" 
          alt="reKro Logo"
          className="h-8 w-auto mr-4">
          </img>
      </div>
      <div className="h-1/5"></div>
      <div className="w-1/2">
        <p className="text-bold text-3xl text-center mb-6">
          What can I help you with?
        </p>

        <div className="grid grid-cols-2 gap-4 w-auto">
          {options.map((option, index) => (
            <button
              key={index}
              className="flex flex-row text-sm items-center gap-0.5 
                      bg-white shadow-md px-1 py-3 rounded-lg w-full h-auto 
                      text-gray-700 font-medium hover:bg-gray-100 transition 
                      break-words cursor-pointer"
              onClick={() => setInput(option.text)}
            >
              <span className="text-xl">{option.icon}</span>
              <span className="text-left flex-wrap whitespace-normal leading-none">
              {option.text}
            </span>
            </button>
          ))}
        </div>

        <div className="bg-white flex flex-col items-center mt-4 shadow-lg 
        border-[1px] border-gray-300 h-32 rounded-lg">
          <textarea 
            className="bg-white w-full rounded-lg p-3 h-30 focus:outline-none"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          >
          </textarea>
          <div className="flex flex-row items-center justify-between w-full
            h-12 mb-2 bg-white">
            <div>
              {/* <div className="flex flex-row items-center justify-center rounded-lg
              border-[1px] px-2 py-1 ml-2 cursor-pointer border-gray-300">
                <AddReactionIcon></AddReactionIcon>
              </div> */}
            </div>
            <div className="flex items-center justify-center mr-4
            p-1 rounded-full bg-[#244E48] cursor-pointer"
            onClick={handleSubmit}
            >
              <ArrowUpwardIcon className="text-white"></ArrowUpwardIcon>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
