import axios from 'axios';
import { CreateMessage } from '@/db';

export async function POST(req: Request) {
  const { messages, model, chat_id, chat_user_id } = await req.json();
  
  // Fetch the last user message
  const lastMessage = messages[messages.length - 1];
  
  // Saving user messages to the database
  await CreateMessage(chat_id, lastMessage.content, lastMessage.role);
  
  // Calling Project Answer Generation Interface
  let answer;
  try {
    const response = await axios.post('http://rekro-backend:5001/api/generate-answer', {
      query: lastMessage.content,
      top_k: 12,
      context_size: 5,
      model: 'qwen-plus' 
    });
    
    // Answers returned by projects are typically in response.data.answer
    answer = response.data.answer;
  } catch (error) {
    console.error('Error when calling project answer generation interface:', error);
    return new Response('Error generating answer', { status: 500 });
  }
  
  // Saving assistant replies to the database
  await CreateMessage(chat_id, answer, 'assistant');
  
  // Return to Generated Answers
  return new Response(answer, { headers: { 'Content-Type': 'text/plain' } });
}
