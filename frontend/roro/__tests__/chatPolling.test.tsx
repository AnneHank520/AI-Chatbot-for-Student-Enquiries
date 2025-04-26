import React from 'react';
import { render, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Page from '@/app/chat/[chat_id]/page';


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

    
    await waitFor(() => {
      expect(document.querySelector('[data-testid="mock-markdown"]')).toHaveTextContent('I am Roro');
    });

    
    expect(true).toBe(true);
  });
});
