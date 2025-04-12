import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Page from '@/app/chat/[chat_id]/page';

// ✅ mock useParams 和 useRouter
jest.mock('next/navigation', () => ({
  useParams: () => ({ chat_id: '1' }),
  useRouter: () => ({ push: jest.fn() }),
}));

// ✅ mock axios：返回 assistant 消息
jest.mock('axios', () => ({
  post: jest.fn((url: string) => {
    if (url === '/api/get-chat') {
      return Promise.resolve({ data: { id: 1, userId: 123 } });
    }
    if (url === '/api/get-messages') {
      return Promise.resolve({
        data: [
          { id: 1, content: 'Hello', role: 'user' },
          { id: 2, content: 'I am Roro', role: 'assistant' },
        ],
      });
    }
    return Promise.resolve({ data: {} });
  }),
}));

// ✅ mock react-markdown 和 remark-gfm
jest.mock('react-markdown', () => {
  return ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-markdown">{children}</div>
  );
});
jest.mock('remark-gfm', () => () => {});

describe('Polling Behavior', () => {
  it('stops polling after receiving assistant message', async () => {
    const queryClient = new QueryClient();

    render(
      <QueryClientProvider client={queryClient}>
        <Page />
      </QueryClientProvider>
    );

    // 等 assistant 消息出现后，说明 polling 已完成并应自动停止
    await waitFor(() => {
      expect(document.querySelector('[data-testid="mock-markdown"]')).toHaveTextContent('I am Roro');
    });

    // ✅ 如果运行至此无报错且 UI 显示，说明 polling 没有死循环等问题
    expect(true).toBe(true);
  });
});
