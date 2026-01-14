import { render, screen } from '@testing-library/react'
import Contact from '@/components/Contact'

describe('Contact Component', () => {
  it('renders the contact section', () => {
    render(<Contact />)

    expect(screen.getByText(/get in touch/i)).toBeInTheDocument()
  })

  it('displays contact section heading', () => {
    render(<Contact />)

    expect(screen.getByRole('heading', { name: /get in touch/i })).toBeInTheDocument()
  })

  it('displays description text', () => {
    render(<Contact />)

    expect(screen.getByText(/interested in discussing new opportunities/i)).toBeInTheDocument()
  })

  it('renders LinkedIn profile link', () => {
    render(<Contact />)

    const linkedInLink = screen.getByRole('link', { name: /linkedin connect professionally/i })
    expect(linkedInLink).toHaveAttribute(
      'href',
      'https://www.linkedin.com/in/gustavo-tsuji-7100462b'
    )
    expect(linkedInLink).toHaveAttribute('target', '_blank')
    expect(linkedInLink).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('renders GitHub profile link', () => {
    render(<Contact />)

    const githubLink = screen.getByRole('link', { name: /github check out my code/i })
    expect(githubLink).toHaveAttribute('href', 'https://github.com/gustavotsuji')
    expect(githubLink).toHaveAttribute('target', '_blank')
    expect(githubLink).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('has proper section structure', () => {
    const { container } = render(<Contact />)

    const section = container.querySelector('section')
    expect(section).toBeInTheDocument()
    expect(section).toHaveAttribute('id', 'contact')
  })

  it('displays social icons', () => {
    const { container } = render(<Contact />)

    const svgs = container.querySelectorAll('svg')
    expect(svgs.length).toBeGreaterThanOrEqual(2) // LinkedIn and GitHub icons
  })
})
