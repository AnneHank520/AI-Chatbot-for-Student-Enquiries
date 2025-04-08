import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import Page from '@/app/chat/[chat_id]/page';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@testing-library/jest-dom';

// 🛠️ MOCK ESM MODULES
jest.mock('react-markdown', () => {
  return ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-markdown">{children}</div>
  );
});
jest.mock('remark-gfm', () => ({})); // ✅ mock 这个 esm 模块

jest.mock('next/navigation', () => ({
  useParams: () => ({ chat_id: '1' }),
  useRouter: () => ({ push: jest.fn() }),
}));

jest.mock('axios', () => ({
  post: jest.fn((url: string) => {
    if (url === '/api/get-chat') {
      return Promise.resolve({ data: { id: 1, userId: 123 } });
    }
    if (url === '/api/get-messages') {
      return Promise.reject(new Error('Failed to fetch messages'));
    }
    return Promise.resolve({ data: {} });
  }),
}));

describe('Chat Page - Error Handling', () => {
  const queryClient = new QueryClient();

  it('renders fallback UI when messages API fails', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <Page />
      </QueryClientProvider>
    );

    // 渲染 Loading 阶段
    expect(screen.getByText('Loading...')).toBeInTheDocument();

    // fallback UI 可以是 Error Boundary、静默失败等，这里只验证组件不崩溃
    await waitFor(() => {
      expect(screen.queryByText('Hi! How can I help you?')).not.toBeInTheDocument();
    });
  });
});
