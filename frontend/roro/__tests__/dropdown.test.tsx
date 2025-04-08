import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Dropdown } from '../src/app/page';

describe('Dropdown Component', () => {
  const mockOnAction = jest.fn();

  const sampleProps = {
    icon: '🎓',
    text: 'Studying in Australia',
    detail: ['What are the benefits of studying in Australia?'],
    onAction: mockOnAction,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders icon and text', () => {
    render(<Dropdown {...sampleProps} />);
    expect(screen.getByText('🎓')).toBeInTheDocument();
    expect(screen.getByText('Studying in Australia')).toBeInTheDocument();
  });

  it('shows dropdown items on hover', async () => {
    render(<Dropdown {...sampleProps} />);
    const button = screen.getByText('Studying in Australia');
    fireEvent.mouseEnter(button);

    await waitFor(() => {
      expect(screen.getByText(sampleProps.detail[0])).toBeInTheDocument();
    });
  });

  it('hides dropdown items on mouse leave', async () => {
    render(<Dropdown {...sampleProps} />);
    const button = screen.getByText('Studying in Australia');

    fireEvent.mouseEnter(button);
    await screen.findByText(sampleProps.detail[0]);

    fireEvent.mouseLeave(button);

    await waitFor(() => {
      expect(screen.queryByText(sampleProps.detail[0])).not.toBeInTheDocument();
    });
  });

  it('calls onAction when item clicked', async () => {
    render(<Dropdown {...sampleProps} />);
    fireEvent.mouseEnter(screen.getByText(sampleProps.text));

    const item = await screen.findByText(sampleProps.detail[0]);
    fireEvent.click(item);

    expect(mockOnAction).toHaveBeenCalledTimes(1);
    expect(mockOnAction).toHaveBeenCalledWith(sampleProps.detail[0]);
  });
});
