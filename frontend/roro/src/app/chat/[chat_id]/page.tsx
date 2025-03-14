'use client';

import { useChat } from '@ai-sdk/react';
import { useEffect, useRef, useState } from 'react';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import { useParams } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
// import AddReactionIcon from '@mui/icons-material/AddReaction';

export default function Page() {

  const {chat_id} = useParams()

  const{data: chat} = useQuery({
    queryKey: ['chat', chat_id],
    queryFn: () => {
      return axios.post(`/api/get-chat`, {
        chat_id: chat_id
      })
    }
  })

  const{data: preMessages} = useQuery({
    queryKey: ['messages', chat_id],
    queryFn: () => {
      return axios.post(`/api/get-messages`, {
        chat_id: chat_id,
        chat_user_id: chat?.data?.userId
      })
    },
    enabled: !!chat?.data?.id
  })

  const [model, setModel] = useState("deepseek")

  const { messages, input, handleInputChange, handleSubmit, append } = useChat({
    body: {
      model: model,
      chat_id: chat_id,
      chat_user_id: chat?.data?.userId
    },
    initialMessages: preMessages?.data
  });

  const endRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (endRef.current) {
        endRef?.current?.scrollIntoView({behavior: 'smooth'})
    }
  }, [messages])

  const handleFirstMessage = async() => {
    if (chat?.data?.title && preMessages?.data?.length === 0) {
      await append({
        role: 'user',
        content: chat?.data?.title
      }), {
        model: model,
        chat_id: chat_id,
        chat_user_id: chat?.data?.userId
      }
    }
  }

  useEffect(() => {
    handleFirstMessage()
  }, [chat?.data?.title, preMessages])

  return (
    <div className='flex flex-col h-full justify-between items-center'>
      <div className="flex items-center justify-center">
          <img
          src="/images/Logo Name.svg" 
          alt="reKro Logo"
          className="h-8 w-auto mr-4">
          </img>
      </div>
      <div className='flex flex-col h-full w-full justify-between items-center overflow-y-auto'>
        <div className='flex flex-col w-2/3 gap-8 justify-between flex-1'>
          <div className='h-4'></div>
          <div className='flex flex-col gap-8 flex-1'>
              {messages?.map(message => (
                  <div 
                      key={message.id}
                      className={`rounded-lg flex flex-row gap-2 ${message?.role 
                      === 'assistant' ? 'justify-start': 'justify-end'}`}
                  >
                    {/* if it's assistant，make it in left */}
                    {message.role === 'assistant' && (
                      <div className="flex-shrink-0 h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center text-white font-bold">
                        R
                      </div>
                    )}

                    <p className={`inline-block p-2 rounded-lg ${message?.role
                    === 'assistant' ? 'bg-white': 'bg-[#5ABEB0]'}`}>
                        {message?.content}
                    </p>

                    {/* if it's user，make it in right */}
                    {message.role !== 'assistant' && (
                      <div className="flex-shrink-0 h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold">
                        U
                      </div>
                    )}
                  </div>
              ))}
          </div>
        </div>
        <div className='h-4' ref={endRef}>

        </div>
      </div>

        {/*输入框*/}
      <div className="bg-white flex flex-col items-center mt-4 shadow-lg 
        border-[1px] border-gray-300 h-32 rounded-lg w-2/3">
          <textarea 
            className="bg-white w-full rounded-lg p-3 h-30 focus:outline-none"
            value={input}
            onChange={handleInputChange}
          >
          </textarea>
          <div className="flex flex-row items-center justify-between w-full
            h-12 mb-2 bg-white">
            <div>
              {/* <div className={`flex flex-row items-center justify-center rounded-lg
              border-[1px] px-2 py-1 ml-2 cursor-pointer border-gray-300"}`}>
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
  );
}