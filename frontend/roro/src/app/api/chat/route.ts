import { CreateMessage } from '@/db';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { streamText } from 'ai';

// Allow streaming responses up to 20 seconds
export const maxDuration = 20;

const qwen_turbo = createDeepSeek({
    apiKey: process.env.Try_API_KEY,
    baseURL: process.env.BASE_URL
})

export async function POST(req: Request) {
  const { messages, model, chat_id, chat_user_id } = await req.json();

  // deposit user's messages to the database
  const lastMessage = messages[messages.length - 1]
  await CreateMessage(chat_id, lastMessage.content, lastMessage.role)

  const result = streamText({
    model: qwen_turbo('qwen-turbo'),
    system: 'You are a helpful assistant.',
    messages,
    onFinish: async (result) => {
      await CreateMessage(chat_id, result.text, 'assistant')
    }
  });

  return result.toDataStreamResponse();
}