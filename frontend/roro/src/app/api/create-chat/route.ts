import { CreateChat, getChats } from "@/db";
import { cookies } from "next/headers";
import crypto from "crypto";

export async function POST(req: Request) {
    const {title, model} = await req.json()

    let userId = (await cookies()).get("userId")?.value;

    if (!userId) {
        userId = crypto.randomUUID();
        (await cookies()).set("userId", userId, { maxAge: 60 * 60 * 24 * 7 }); // 7 天有效
    }

    const existingChats = await getChats(userId);
    // 假设对于同一 model，我们只保留一条聊天记录
    const existingChat = existingChats?.find(chat => chat.model === model);
    if (existingChat) {
        // 如果存在，直接返回已有聊天 id
        return new Response(JSON.stringify({ id: existingChat.id }), { status: 200 });
    }

    // 如果不存在，则创建新聊天
    const newChat = await CreateChat(title, userId, model);
    return new Response(JSON.stringify({ id: newChat?.id }), { status: 200 });
}

