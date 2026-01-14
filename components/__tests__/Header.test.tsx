import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Header from '@/components/Header'

describe('Header Component', () => {
  it('renders the header with navigation', () => {
    render(<Header />)

    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(screen.getByRole('navigation')).toBeInTheDocument()
  })

  it('displays the site logo/title', () => {
    render(<Header />)

    const logo = screen.getByRole('link', { name: 'GT' })
    expect(logo).toBeInTheDocument()
    expect(logo).toHaveAttribute('href', '/')
  })

  it('renders desktop navigation links', () => {
    render(<Header />)

    // Get all links with specific text (desktop version)
    const aboutLinks = screen.getAllByRole('link', { name: /about/i })
    const blogLinks = screen.getAllByRole('link', { name: /blog/i })
    const contactLinks = screen.getAllByRole('link', { name: /contact/i })

    expect(aboutLinks.length).toBeGreaterThan(0)
    expect(blogLinks.length).toBeGreaterThan(0)
    expect(contactLinks.length).toBeGreaterThan(0)
  })

  it('has correct href attributes for navigation links', () => {
    render(<Header />)

    const aboutLink = screen.getAllByRole('link', { name: /about/i })[0]
    const blogLink = screen.getAllByRole('link', { name: /blog/i })[0]
    const contactLink = screen.getAllByRole('link', { name: /contact/i })[0]

    expect(aboutLink).toHaveAttribute('href', '/about')
    expect(blogLink).toHaveAttribute('href', '/blog')
    expect(contactLink).toHaveAttribute('href', '/#contact')
  })

  it('renders mobile menu button', () => {
    render(<Header />)

    const menuButton = screen.getByRole('button', { name: /toggle menu/i })
    expect(menuButton).toBeInTheDocument()
  })

  it('toggles mobile menu when button is clicked', async () => {
    const user = userEvent.setup()
    render(<Header />)

    const menuButton = screen.getByRole('button', { name: /toggle menu/i })

    // Initially, mobile menu links should only be desktop (1 of each)
    expect(screen.getAllByRole('link', { name: /about/i }).length).toBe(1)

    // Click to open menu
    await user.click(menuButton)

    // Now there should be both desktop and mobile versions (2 of each)
    expect(screen.getAllByRole('link', { name: /about/i }).length).toBe(2)
  })

  it('has proper semantic HTML structure', () => {
    const { container } = render(<Header />)

    const header = container.querySelector('header')
    const nav = container.querySelector('nav')

    expect(header).toBeInTheDocument()
    expect(nav).toBeInTheDocument()
  })
})
