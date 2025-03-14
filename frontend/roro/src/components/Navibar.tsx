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
        <div className='h-screen bg-gray-50'>
            <div className="flex items-center justify-center">
                <p className="font-bold text-2xl">
                    roro
                </p>
            </div>

            <div className='h-10 flex items-center justify-center mt-4
                cursor-pointer'
                onClick={() => {
                    router.push('/')
                }}
            >
                <p className='inline-block bg-blue-100 rounded-lg px-2 py-1 font-thin'>
                    Create New Session
                </p>
            </div>

            {/* catalogue */}
            <div className='flex flex-col items-center justify-center gap-2 p-6'>
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
            </div>
        </div>
    )
}

export default Navibar