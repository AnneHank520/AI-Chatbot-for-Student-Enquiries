import React from 'react'; 
import { render, screen } from '@testing-library/react';

describe('Sanity Test', () => {
  it('renders basic text', () => {
    render(<div>Hello Test!</div>);
    expect(screen.getByText('Hello Test!')).toBeInTheDocument();
  });
});

