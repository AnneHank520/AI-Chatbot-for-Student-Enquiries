import React from 'react';

export default function ReactMarkdown({ children }: { children: React.ReactNode }) {
  return <div data-testid="mock-markdown">{children}</div>;
}
