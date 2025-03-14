import { CreateChat } from "@/db";
import { cookies } from "next/headers";
import crypto from "crypto";

export async function POST(req: Request) {
    const {title, model} = await req.json()

    let userId = (await cookies()).get("userId")?.value;

    if (!userId) {
        userId = crypto.randomUUID();
        (await cookies()).set("userId", userId, { maxAge: 60 * 60 * 24 * 7 }); // 7 天有效
    }
    // 1. Create a chat
    const newChat = await CreateChat(title, userId, model)

    // 2. Return the chatId
    return new Response(JSON.stringify({id: newChat?.id}), {status: 200})

}