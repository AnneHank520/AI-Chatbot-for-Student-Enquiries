// __tests__/homeError.test.tsx
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import Home from '@/app/page';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@testing-library/jest-dom';

jest.mock('next/navigation', () => ({
  useRouter: () => ({ push: jest.fn() }),
}));

jest.mock('axios', () => ({
  post: jest.fn((url) => {
    if (url === '/api/get-messages') {
      return Promise.reject(new Error('Failed to fetch messages'));
    }
    if (url === '/api/create-chat') {
      return Promise.resolve({ data: { id: 1 } });
    }
    if (url === '/api/chat') {
      return Promise.resolve({ data: { success: true } });
    }
    return Promise.resolve({ data: {} });
  }),
}));

describe('Home Page Error Handling', () => {
  const queryClient = new QueryClient();

  it('renders fallback UI when /api/get-messages fails', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <Home />
      </QueryClientProvider>
    );

    // 默认没有报错边界时应能正常渲染 dropdown
    expect(await screen.findByText('What can I help you with?')).toBeInTheDocument();
  });
});
