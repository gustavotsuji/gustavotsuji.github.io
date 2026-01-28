import { render, screen } from '@testing-library/react'

// Mock getAllPosts before importing the module that uses it so the server component
// receives the mocked data at import time.
jest.mock('@/lib/posts', () => ({
  getAllPosts: jest.fn(() => [
    {
      // include language suffix so BlogLanguageSelector (which filters by `.en`) picks them
      slug: 'test-post-1.en',
      title: 'Test Post 1',
      // match expected displayed dates in tests
      date: '2026-01-09',
      excerpt: 'This is the first test post excerpt',
      tags: ['testing', 'jest'],
      author: 'Gustavo Tsuji',
    },
    {
      slug: 'test-post-2.en',
      title: 'Test Post 2',
      date: '2026-01-10',
      excerpt: 'This is the second test post excerpt',
      tags: ['react', 'nextjs'],
      author: 'Gustavo Tsuji',
    },
  ]),
}))

import BlogPage, { metadata } from '../page'

describe('BlogPage', () => {
  it('renders without crashing', () => {
    render(<BlogPage />)
    expect(screen.getByRole('heading', { name: /^blog$/i, level: 1 })).toBeInTheDocument()
  })

  it('renders main heading', () => {
    render(<BlogPage />)
    const heading = screen.getByRole('heading', { name: /^blog$/i, level: 1 })
    expect(heading).toBeInTheDocument()
  })

  it('renders description text', () => {
    render(<BlogPage />)
    const descriptions = screen.getAllByText(
      /thoughts on software engineering, cloud architecture/i
    )
    // header and preview copy both include this phrase; ensure at least one exists
    expect(descriptions.length).toBeGreaterThanOrEqual(1)
  })

  it('renders all blog posts', () => {
    render(<BlogPage />)
    expect(screen.getByText('Test Post 1')).toBeInTheDocument()
    expect(screen.getByText('Test Post 2')).toBeInTheDocument()
  })

  it('renders post excerpts', () => {
    render(<BlogPage />)
    expect(screen.getByText('This is the first test post excerpt')).toBeInTheDocument()
    expect(screen.getByText('This is the second test post excerpt')).toBeInTheDocument()
  })

  it('renders post tags', () => {
    render(<BlogPage />)
    expect(screen.getByText('testing')).toBeInTheDocument()
    expect(screen.getByText('jest')).toBeInTheDocument()
    expect(screen.getByText('react')).toBeInTheDocument()
    expect(screen.getByText('nextjs')).toBeInTheDocument()
  })

  it('renders post dates in correct format', () => {
    const { container } = render(<BlogPage />)
    // check raw datetime attributes to avoid timezone/localization differences
    expect(container.querySelector('time[datetime="2026-01-09"]')).toBeInTheDocument()
    expect(container.querySelector('time[datetime="2026-01-10"]')).toBeInTheDocument()
  })

  it('renders read more links for each post', () => {
    render(<BlogPage />)
    // BlogPreview now renders "Read more →" links
    const readMoreLinks = screen.getAllByText(/read more/i)
    expect(readMoreLinks).toHaveLength(2)
  })

  it('has correct link to first post', () => {
    render(<BlogPage />)
    const link = screen.getByRole('link', { name: /test post 1/i })
    expect(link).toHaveAttribute('href', '/blog/en/2026/01/test-post-1')
  })

  it('has correct link to second post', () => {
    render(<BlogPage />)
    const link = screen.getByRole('link', { name: /test post 2/i })
    expect(link).toHaveAttribute('href', '/blog/en/2026/01/test-post-2')
  })

  it('renders articles with proper semantic HTML', () => {
    const { container } = render(<BlogPage />)
    const articles = container.querySelectorAll('article')
    expect(articles).toHaveLength(2)
  })

  it('has correct container structure', () => {
    const { container } = render(<BlogPage />)
    const mainDiv = container.querySelector('.min-h-screen')
    expect(mainDiv).toBeInTheDocument()
  })

  it('renders with proper spacing', () => {
    const { container } = render(<BlogPage />)
    const mainDiv = container.querySelector('.pt-24')
    expect(mainDiv).toBeInTheDocument()
  })

  it('renders posts in articles with border', () => {
    const { container } = render(<BlogPage />)
    const articles = container.querySelectorAll('article')
    articles.forEach((article, index) => {
      if (index < articles.length - 1) {
        // articles now use rounded cards; assert they have the card class
        expect(article.className).toContain('rounded-lg')
      }
    })
  })
})

describe('BlogPage Metadata', () => {
  it('has correct title', () => {
    expect(metadata.title).toBe('Blog - Gustavo Tsuji')
  })

  it('has correct description', () => {
    expect(metadata.description).toContain('Technical articles')
    expect(metadata.description).toContain('software engineering')
    expect(metadata.description).toContain('cloud architecture')
  })
})
