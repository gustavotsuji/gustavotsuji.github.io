import { render, screen } from '@testing-library/react'
import Home from '../page'

// Mock dos componentes
jest.mock('@/components/Hero', () => {
  return function MockHero() {
    return <div data-testid="hero">Hero Component</div>
  }
})

jest.mock('@/components/BlogPreview', () => {
  return function MockBlogPreview() {
    return <div data-testid="blog-preview">BlogPreview Component</div>
  }
})

jest.mock('@/components/Contact', () => {
  return function MockContact() {
    return <div data-testid="contact">Contact Component</div>
  }
})

describe('Home Page', () => {
  it('renders without crashing', () => {
    render(<Home />)
    expect(screen.getByTestId('hero')).toBeInTheDocument()
  })

  it('renders Hero component', () => {
    render(<Home />)
    expect(screen.getByTestId('hero')).toBeInTheDocument()
    expect(screen.getByText('Hero Component')).toBeInTheDocument()
  })

  it('renders BlogPreview component', () => {
    render(<Home />)
    expect(screen.getByTestId('blog-preview')).toBeInTheDocument()
    expect(screen.getByText('BlogPreview Component')).toBeInTheDocument()
  })

  it('renders Contact component', () => {
    render(<Home />)
    expect(screen.getByTestId('contact')).toBeInTheDocument()
    expect(screen.getByText('Contact Component')).toBeInTheDocument()
  })

  it('renders all three main sections', () => {
    render(<Home />)
    expect(screen.getByTestId('hero')).toBeInTheDocument()
    expect(screen.getByTestId('blog-preview')).toBeInTheDocument()
    expect(screen.getByTestId('contact')).toBeInTheDocument()
  })

  it('renders sections in correct order', () => {
    const { container } = render(<Home />)
    const sections = container.querySelectorAll('[data-testid]')
    expect(sections[0]).toHaveAttribute('data-testid', 'hero')
    expect(sections[1]).toHaveAttribute('data-testid', 'blog-preview')
    expect(sections[2]).toHaveAttribute('data-testid', 'contact')
  })
})
