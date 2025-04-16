import { CreateMessage } from "@/db";

  
export async function POST(req: Request) {
  try {
    // Get the chatId, content and role from the request body.
    const { chatId, content, role } = await req.json();

    // Call the CreateMessage function to insert a new message into the database.
    const newMessage = await CreateMessage(chatId, content, role);

    if (!newMessage) {
      return new Response(
        JSON.stringify({ error: "Error creating message" }),
        { status: 500 }
      );
    }

    // Return new message data
    return new Response(JSON.stringify(newMessage), { status: 200 });
  } catch (error) {
    console.error("Error in /api/create-message", error);
    return new Response(
      JSON.stringify({ error: "Internal Server Error" }),
      { status: 500 }
    );
  }



}
