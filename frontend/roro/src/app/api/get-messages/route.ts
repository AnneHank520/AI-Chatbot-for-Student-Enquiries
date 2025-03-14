import { getMessagesByChatId } from "@/db"

export async function POST(req: Request) {
    const {chat_id, chat_user_id} = await req.json()

    const messages = await getMessagesByChatId(chat_id)
    return new Response(JSON.stringify(messages), {status: 200})
}