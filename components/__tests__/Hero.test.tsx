import { render, screen } from '@testing-library/react'
import Hero from '@/components/Hero'

describe('Hero Component', () => {
  it('renders the hero section with correct name', () => {
    render(<Hero />)

    expect(screen.getByText('Gustavo')).toBeInTheDocument()
    expect(screen.getByText('Tsuji')).toBeInTheDocument()
  })

  it('displays the job title', () => {
    render(<Hero />)

    expect(screen.getByText('Senior Software Engineer')).toBeInTheDocument()
  })

  it('shows the professional summary', () => {
    render(<Hero />)

    expect(screen.getByText(/18\+ years building scalable backend systems/i)).toBeInTheDocument()
  })

  it('renders call-to-action buttons', () => {
    render(<Hero />)

    const getInTouchButton = screen.getByRole('link', { name: /get in touch/i })
    const blogButton = screen.getByRole('link', { name: /read my blog/i })

    expect(getInTouchButton).toBeInTheDocument()
    expect(getInTouchButton).toHaveAttribute('href', '/#contact')

    expect(blogButton).toBeInTheDocument()
    expect(blogButton).toHaveAttribute('href', '/blog')
  })

  it('has correct styling classes for gradient background', () => {
    const { container } = render(<Hero />)

    const section = container.querySelector('section')
    expect(section).toHaveClass('bg-gradient-to-br')
  })

  it('renders scroll indicator', () => {
    render(<Hero />)

    const scrollLink = screen.getByRole('link', { name: /scroll to blog section/i })
    expect(scrollLink).toBeInTheDocument()
    expect(scrollLink).toHaveAttribute('href', '/#blog')
  })
})
