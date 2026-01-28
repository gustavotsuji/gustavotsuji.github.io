import { render, screen } from '@testing-library/react'
import React from 'react'
import About from '../About'

describe('About', () => {
  it('renders heading and key skills', () => {
    render(<About />)
    const heading = screen.getByRole('heading', { name: /about me/i })
    expect(heading).toBeInTheDocument()
    // check a known skill exists
    expect(screen.getByText(/TypeScript/)).toBeInTheDocument()
  })
})
