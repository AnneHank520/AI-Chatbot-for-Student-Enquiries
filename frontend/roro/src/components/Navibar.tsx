'use client'

import { ChatModel } from '@/db/schema'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { usePathname, useRouter } from 'next/navigation'
import React from 'react'

type Props = {}

const Navibar = (props: Props) => {

    // const user = {id: userId}
    const router = useRouter()

    const {data: chats} = useQuery({
        queryKey: ['chats'],
        queryFn: () => {
            return axios.post('/api/get-chats')
        } 
    })

    const pathname = usePathname()

    return (
        <div className='h-screen w-full flex flex-col items-end'>
            <div className='flex flex-col w-129 flex-1 my-7'>
                {/* Requests */}
                <div className="mb-6">
                    <h2 className="text-2xl font-semibold text-[#183728] mb-2">Requests</h2>
                    <div className="bg-white text-gray-300 text-xl text-center p-3 rounded-2xl">
                    No friend requests
                    </div>
                </div>

                {/* Inbox */}
                <div>
                    <h2 className="text-2xl font-semibold text-[#183728] mb-2">Inbox</h2>
                    {/* Outer container with slight shadow/rounded corners that wraps around the contact list */}
                    <div className="bg-white rounded-2xl shadow-sm p-3" 
                    onClick={() => {
                    router.push('/')
                    }}>
                        <ul className="space-y-2">
                            <li className="flex items-center space-x-3 p-3 rounded-2xl cursor-pointer border">
                                {/* avatar */}
                                <div className="flex-shrink-0 h-18 w-18 rounded-full bg-gray-300 
                                flex items-center justify-center text-white font-bold">
                                R
                                </div>

                                {/* user information */}
                                <div>
                                    <p className="text-3xl">roro</p>
                                    <p className="text-sm text-gray-500">ChatbotAI of reKro</p>
                                </div>
                            </li>
                        </ul>
                    </div>
                </div>

            {/* catalogue */}
            {/* <div className='flex flex-col items-center justify-center gap-2 p-6'>
                {chats?.data?.map((chat: ChatModel) => (
                    <div className='w-full h-10'
                         key={chat.id}
                         onClick={() => {
                            router.push(`/chat/${chat.id}`)
                         }}
                    >
                        <p className={`font-extralight text-sm line-clamp-1 cursor-pointer 
                        bg-[#EDF7F1] ${pathname === `/chat/${chat.id}` ? 'text-blue-700' : ''}`}>
                            {chat?.title}
                        </p>
                    </div>
                ))}
            </div> */}
            </div>
        </div>
    )
}

export default Navibar
