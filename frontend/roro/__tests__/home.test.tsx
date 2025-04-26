import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Home from '@/app/page';

// 🔧 mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({ push: jest.fn() }),
  useParams: () => ({}),
}));

// 🔧 mock react-markdown
jest.mock('react-markdown', () => {
  return ({ children }: { children: React.ReactNode }) => (
    <div data-testid="mock-markdown">{children}</div>
  );
});

// 🔧 mock axios
jest.mock('axios', () => {
  const mockPost = jest.fn((url: string) => {
    if (url === '/api/create-chat') {
      return Promise.resolve({ data: { id: 999 } });
    }
    if (url === '/api/chat') {
      return Promise.resolve({ data: {} });
    }
    if (url === '/api/get-messages') {
      return Promise.resolve({ data: [] }); // ✅ 防止 map 报错
    }
    return Promise.resolve({ data: {} });
  });

  return {
    __esModule: true,
    default: {
      post: mockPost,
    },
    post: mockPost,
  };
});

describe('Home Page Integration', () => {
  const queryClient = new QueryClient();

  const renderHome = () =>
    render(
      <QueryClientProvider client={queryClient}>
        <Home />
      </QueryClientProvider>
    );

  it('renders dropdown categories', () => {
    renderHome();
    expect(screen.getByText('Studying in Australia')).toBeInTheDocument();
    expect(screen.getByText('Preparing for Your Journey')).toBeInTheDocument();
  });

  it('triggers chat creation when dropdown detail is clicked', async () => {
    renderHome();

    
    const button = screen.getByText('Studying in Australia');
    fireEvent.mouseEnter(button);

    const item = await screen.findByText('What are the benefits of studying in Australia?');
    fireEvent.click(item);

    
    await waitFor(() => {
      expect(screen.getByRole('textbox')).toHaveValue('');
    });
  });
});

