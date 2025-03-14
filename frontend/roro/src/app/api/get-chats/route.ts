import { getChats } from "@/db";
import { cookies } from "next/headers";
import crypto from "crypto";

export async function POST(req: Request) {
    let userId = (await cookies()).get("userId")?.value;

    if (!userId) {
        userId = crypto.randomUUID();
        (await cookies()).set("userId", userId, { maxAge: 60 * 60 * 24 * 7 }); 
    }

    const chats = await getChats(userId)

    return new Response(JSON.stringify(chats), {status: 200})


}