import axios from 'axios';
import { CreateMessage } from '@/db';

export async function POST(req: Request) {
  const { messages, model, chat_id, chat_user_id } = await req.json();
  
  // 取出最后一条用户消息
  const lastMessage = messages[messages.length - 1];
  
  // 保存用户消息到数据库
  await CreateMessage(chat_id, lastMessage.content, lastMessage.role);
  
  // 调用旧项目回答生成接口
  let answer;
  try {
    const response = await axios.post('http://rekro-backend:5001/api/generate-answer', {
      query: lastMessage.content,
      top_k: 12,
      context_size: 5,
      model: 'qwen-plus' // 根据需要传入旧项目能识别的模型名称
    });
    
    // 旧项目返回的回答一般在 response.data.answer 中
    answer = response.data.answer;
  } catch (error) {
    console.error('调用旧项目回答生成接口时出错:', error);
    return new Response('Error generating answer', { status: 500 });
  }
  
  // 保存助手回复到数据库
  await CreateMessage(chat_id, answer, 'assistant');
  
  // 返回生成的回答
  return new Response(answer, { headers: { 'Content-Type': 'text/plain' } });
}
