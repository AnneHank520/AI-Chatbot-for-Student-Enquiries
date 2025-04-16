import { CreateChat, getChats } from "@/db";
import { cookies } from "next/headers";
import crypto from "crypto";

export async function POST(req: Request) {
    const {title, model} = await req.json()

    let userId = (await cookies()).get("userId")?.value;

    if (!userId) {
        userId = crypto.randomUUID();
        (await cookies()).set("userId", userId, { maxAge: 60 * 60 * 24 * 7 }); 
    }

    const existingChats = await getChats(userId);
    // Let's assume that for the same model, we keep only one chat record
    const existingChat = existingChats?.find(chat => chat.model === model);
    if (existingChat) {
        // If it exists, return the existing chat id directly
        return new Response(JSON.stringify({ id: existingChat.id }), { status: 200 });
    }

    // If it doesn't exist, create a new chat
    const newChat = await CreateChat(title, userId, model);
    return new Response(JSON.stringify({ id: newChat?.id }), { status: 200 });
}

