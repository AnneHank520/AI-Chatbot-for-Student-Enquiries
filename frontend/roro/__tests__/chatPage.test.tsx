import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Page from '@/app/chat/[chat_id]/page';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// mock useParams useRouter
jest.mock('next/navigation', () => ({
  useParams: () => ({ chat_id: '1' }),
  useRouter: () => ({ push: jest.fn() }),
}));

// mock axios
jest.mock('axios', () => ({
  post: jest.fn((url, payload) => {
    if (url === '/api/get-chat') {
      return Promise.resolve({ data: { id: 1, userId: 123 } });
    }
    if (url === '/api/get-messages') {
      return Promise.resolve({
        data: [
          { id: 1, content: 'Hello', role: 'user' },
          { id: 2, content: 'Hi! How can I help you?', role: 'assistant' },
        ],
      });
    }
    if (url === '/api/chat') {
      return Promise.resolve({ data: { success: true } });
    }
    return Promise.resolve({ data: {} });
  }),
}));

// mock remark-gfm to avoid ESM error
jest.mock('remark-gfm', () => () => {});

// mock react-markdown
jest.mock('react-markdown', () => {
  return ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-markdown">{children}</div>
  );
});

describe('Chat Page', () => {
  const queryClient = new QueryClient();

  const renderPage = () =>
    render(
      <QueryClientProvider client={queryClient}>
        <Page />
      </QueryClientProvider>
    );

  it('renders messages after fetching', async () => {
    renderPage();
    expect(screen.getByText('Loading...')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Hello')).toBeInTheDocument();
      expect(screen.getByTestId('mock-markdown')).toHaveTextContent('Hi! How can I help you?');
    });
  });

  it('allows user to input and send a message', async () => {
    renderPage();

    await screen.findByText('Hello');

    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: "What's up, Roro?" } });

    
    const sendButton = screen.getByLabelText('send');
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(textarea).toHaveValue('');
    });
  });
});

