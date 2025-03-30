import { CreateMessage } from "@/db";

  
export async function POST(req: Request) {
  try {
    // 从请求体中获取 chatId、content 和 role
    const { chatId, content, role } = await req.json();

    // 调用 CreateMessage 函数向数据库插入一条新消息
    const newMessage = await CreateMessage(chatId, content, role);

    if (!newMessage) {
      return new Response(
        JSON.stringify({ error: "Error creating message" }),
        { status: 500 }
      );
    }

    // 返回新消息数据
    return new Response(JSON.stringify(newMessage), { status: 200 });
  } catch (error) {
    console.error("Error in /api/create-message", error);
    return new Response(
      JSON.stringify({ error: "Internal Server Error" }),
      { status: 500 }
    );
  }



}
