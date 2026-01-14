import { render, screen } from '@testing-library/react'
import BlogPage, { metadata } from '../page'

// Mock do getAllPosts
jest.mock('@/lib/posts', () => ({
  getAllPosts: jest.fn(() => [
    {
      slug: 'test-post-1',
      title: 'Test Post 1',
      date: '2026-01-10',
      excerpt: 'This is the first test post excerpt',
      tags: ['testing', 'jest'],
      author: 'Gustavo Tsuji',
    },
    {
      slug: 'test-post-2',
      title: 'Test Post 2',
      date: '2026-01-11',
      excerpt: 'This is the second test post excerpt',
      tags: ['react', 'nextjs'],
      author: 'Gustavo Tsuji',
    },
  ]),
}))

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
    expect(
      screen.getByText(/thoughts on software engineering, cloud architecture/i)
    ).toBeInTheDocument()
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
    render(<BlogPage />)
    expect(screen.getByText('January 9, 2026')).toBeInTheDocument()
    expect(screen.getByText('January 10, 2026')).toBeInTheDocument()
  })

  it('renders read more links for each post', () => {
    render(<BlogPage />)
    const readMoreLinks = screen.getAllByText(/read full article/i)
    expect(readMoreLinks).toHaveLength(2)
  })

  it('has correct link to first post', () => {
    render(<BlogPage />)
    const link = screen.getByRole('link', { name: /test post 1/i })
    expect(link).toHaveAttribute('href', '/blog/test-post-1')
  })

  it('has correct link to second post', () => {
    render(<BlogPage />)
    const link = screen.getByRole('link', { name: /test post 2/i })
    expect(link).toHaveAttribute('href', '/blog/test-post-2')
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
        expect(article.className).toContain('border-b')
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
