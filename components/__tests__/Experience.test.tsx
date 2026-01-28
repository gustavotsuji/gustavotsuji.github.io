import { render, screen } from '@testing-library/react'
import React from 'react'
import Experience from '../Experience'

describe('Experience', () => {
  it('renders Professional Experience heading and company names', () => {
    render(<Experience />)
    expect(screen.getByRole('heading', { name: /professional experience/i })).toBeInTheDocument()
    // check presence of a known company from the list
    expect(screen.getByText(/Grupo OLX/)).toBeInTheDocument()
  })
})
