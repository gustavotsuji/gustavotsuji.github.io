import { render, screen } from '@testing-library/react'
import AboutPage, { metadata } from '../page'

describe('AboutPage', () => {
  it('renders without crashing', () => {
    render(<AboutPage />)
    expect(screen.getByRole('heading', { name: /about me/i })).toBeInTheDocument()
  })

  it('renders main heading', () => {
    render(<AboutPage />)
    const heading = screen.getByRole('heading', { name: /about me/i, level: 1 })
    expect(heading).toBeInTheDocument()
  })

  it('renders introduction text', () => {
    render(<AboutPage />)
    const texts = screen.getAllByText(/18\+ years/i)
    expect(texts.length).toBeGreaterThan(0)
  })

  it('renders profile image', () => {
    render(<AboutPage />)
    const img = screen.getByAltText('Gustavo Tsuji')
    expect(img).toBeInTheDocument()
    expect(img).toHaveAttribute('src', expect.stringContaining('gravatar.com'))
  })

  it('renders contact information section', () => {
    render(<AboutPage />)
    expect(screen.getByText('Contact Information')).toBeInTheDocument()
  })

  it('renders email link', () => {
    render(<AboutPage />)
    const emailLink = screen.getByRole('link', { name: /gustavokt@gmail.com/i })
    expect(emailLink).toBeInTheDocument()
    expect(emailLink).toHaveAttribute('href', 'mailto:gustavokt@gmail.com')
  })

  it('renders LinkedIn link', () => {
    render(<AboutPage />)
    const linkedinLink = screen.getByRole('link', { name: /linkedin/i })
    expect(linkedinLink).toBeInTheDocument()
    expect(linkedinLink).toHaveAttribute('href', expect.stringContaining('linkedin.com'))
  })

  it('renders GitHub link', () => {
    render(<AboutPage />)
    const githubLink = screen.getByRole('link', { name: /github/i })
    expect(githubLink).toBeInTheDocument()
    expect(githubLink).toHaveAttribute('href', expect.stringContaining('github.com'))
  })

  it('renders education section', () => {
    render(<AboutPage />)
    expect(screen.getByText(/education/i)).toBeInTheDocument()
  })

  it('renders USP degree information', () => {
    render(<AboutPage />)
    const uspTexts = screen.getAllByText(/University of São Paulo/i)
    expect(uspTexts.length).toBeGreaterThan(0)
  })

  it('renders professional experience section', () => {
    render(<AboutPage />)
    expect(screen.getByText(/professional experience/i)).toBeInTheDocument()
  })

  it('renders technical skills section', () => {
    render(<AboutPage />)
    expect(screen.getByText(/technical skills/i)).toBeInTheDocument()
  })

  it('has correct container structure', () => {
    const { container } = render(<AboutPage />)
    const mainDiv = container.querySelector('.min-h-screen')
    expect(mainDiv).toBeInTheDocument()
  })

  it('renders with proper spacing classes', () => {
    const { container } = render(<AboutPage />)
    const mainDiv = container.querySelector('.pt-24')
    expect(mainDiv).toBeInTheDocument()
  })
})

describe('AboutPage Metadata', () => {
  it('has correct title', () => {
    expect(metadata.title).toBe('About - Gustavo Tsuji')
  })

  it('has correct description', () => {
    expect(metadata.description).toContain('Senior Software Engineer')
    expect(metadata.description).toContain('18+ years')
  })
})
