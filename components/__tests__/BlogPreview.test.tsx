import { render, screen } from '@testing-library/react'
import BlogPreview from '@/components/BlogPreview'

// Mock the posts data
jest.mock('@/lib/posts', () => ({
  getAllPosts: jest.fn(() => [
    {
      slug: 'test-post-1',
      title: 'Test Post 1',
      date: '2024-01-01',
      excerpt: 'This is a test post excerpt 1',
      tags: ['test', 'jest'],
      author: 'Gustavo Tsuji',
    },
    {
      slug: 'test-post-2',
      title: 'Test Post 2',
      date: '2024-01-02',
      excerpt: 'This is a test post excerpt 2',
      tags: ['testing'],
      author: 'Gustavo Tsuji',
    },
  ]),
}))

describe('BlogPreview Component', () => {
  it('renders the blog preview section', () => {
    render(<BlogPreview />)

    expect(screen.getByText(/latest articles/i)).toBeInTheDocument()
  })

  it('displays blog post titles', () => {
    render(<BlogPreview />)

    expect(screen.getByText('Test Post 1')).toBeInTheDocument()
    expect(screen.getByText('Test Post 2')).toBeInTheDocument()
  })

  it('shows blog post excerpts', () => {
    render(<BlogPreview />)

    expect(screen.getByText(/test post excerpt 1/i)).toBeInTheDocument()
    expect(screen.getByText(/test post excerpt 2/i)).toBeInTheDocument()
  })

  it('renders links to individual blog posts', () => {
    render(<BlogPreview />)

    const post1Link = screen.getByRole('link', { name: /test post 1/i })
    const post2Link = screen.getByRole('link', { name: /test post 2/i })

    expect(post1Link).toHaveAttribute('href', '/blog/test-post-1')
    expect(post2Link).toHaveAttribute('href', '/blog/test-post-2')
  })

  it('displays tags for each post', () => {
    render(<BlogPreview />)

    expect(screen.getByText('test')).toBeInTheDocument()
    expect(screen.getByText('jest')).toBeInTheDocument()
    expect(screen.getByText('testing')).toBeInTheDocument()
  })

  it('shows formatted dates', () => {
    render(<BlogPreview />)

    // Dates should be formatted (adjust based on your date formatting)
    const dateElements = screen.getAllByText(/2024/i)
    expect(dateElements.length).toBeGreaterThan(0)
  })

  it('has a link to view all articles', () => {
    render(<BlogPreview />)

    const viewAllLink = screen.getByRole('link', { name: /view all articles/i })
    expect(viewAllLink).toHaveAttribute('href', '/blog')
  })

  it('has proper section structure', () => {
    const { container } = render(<BlogPreview />)

    const section = container.querySelector('section')
    expect(section).toBeInTheDocument()
    expect(section).toHaveAttribute('id', 'blog')
  })
})
