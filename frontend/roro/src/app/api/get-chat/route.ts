import { getChat } from "@/db";
import { cookies } from "next/headers";
import crypto from "crypto";

export async function POST(req: Request) {
    const { chat_id } = await req.json()

    let userId = (await cookies()).get("userId")?.value;
    
    if (!userId) {
        userId = crypto.randomUUID();
        (await cookies()).set("userId", userId, { maxAge: 60 * 60 * 24 * 7 }); // 7 天有效
    }

    const chat = await getChat(chat_id, userId);
    return new Response(JSON.stringify(chat), {status: 200})
}