import { render, screen } from '@testing-library/react'
import BlogPostCard from '../BlogPostCard'
import type { Post } from '@/lib/posts'

describe('BlogPostCard', () => {
  const mockPost: Post = {
    slug: 'test-post.en',
    title: 'Test Post Title',
    excerpt: 'This is a test excerpt',
    date: '2026-01-15',
    tags: ['testing', 'react'],
    content: 'Test content',
    lang: 'en',
    baseSlug: 'test-post',
  }

  it('renders post title as a link', () => {
    render(<BlogPostCard post={mockPost} />)
    const link = screen.getByRole('link', { name: /test post title/i })
    expect(link).toBeInTheDocument()
  })

  it('renders date in time element', () => {
    const { container } = render(<BlogPostCard post={mockPost} />)
    const timeElement = container.querySelector('time')
    expect(timeElement).toHaveAttribute('dateTime', '2026-01-15')
    expect(timeElement).toHaveTextContent(/2026/)
  })

  it('renders excerpt text', () => {
    render(<BlogPostCard post={mockPost} />)
    expect(screen.getByText(/this is a test excerpt/i)).toBeInTheDocument()
  })

  it('renders all tags', () => {
    render(<BlogPostCard post={mockPost} />)
    expect(screen.getByText('testing')).toBeInTheDocument()
    expect(screen.getByText('react')).toBeInTheDocument()
  })

  it('generates correct href link', () => {
    render(<BlogPostCard post={mockPost} />)
    const link = screen.getByRole('link', { name: /test post title/i })
    expect(link).toHaveAttribute('href', '/blog/en/2026/01/test-post')
  })

  it('renders read more link', () => {
    render(<BlogPostCard post={mockPost} />)
    const readMoreLink = screen.getByRole('link', { name: /read more/i })
    expect(readMoreLink).toBeInTheDocument()
    expect(readMoreLink).toHaveAttribute('href', '/blog/en/2026/01/test-post')
  })

  it('handles post without date gracefully', () => {
    const postWithoutDate = { ...mockPost, date: '' }
    render(<BlogPostCard post={postWithoutDate} />)
    expect(screen.getByText(mockPost.title)).toBeInTheDocument()
    // Should not have read more link if no date
    expect(screen.queryByRole('link', { name: /read more/i })).not.toBeInTheDocument()
  })

  it('handles Portuguese language posts', () => {
    const ptPost: Post = {
      ...mockPost,
      slug: 'test-post.pt',
      lang: 'pt',
      baseSlug: 'test-post',
    }
    render(<BlogPostCard post={ptPost} />)
    const link = screen.getByRole('link', { name: /test post title/i })
    expect(link).toHaveAttribute('href', '/blog/pt/2026/01/test-post')
  })

  it('handles Japanese posts', () => {
    const jaPost: Post = {
      ...mockPost,
      slug: 'test-post.ja',
      lang: 'ja',
      baseSlug: 'test-post',
    }
    render(<BlogPostCard post={jaPost} />)
    const link = screen.getByRole('link', { name: /test post title/i })
    expect(link).toHaveAttribute('href', '/blog/ja/2026/01/test-post')
  })

  it('applies correct CSS classes for styling', () => {
    const { container } = render(<BlogPostCard post={mockPost} />)
    const article = container.querySelector('article')
    expect(article).toHaveClass('bg-gray-50', 'dark:bg-gray-800', 'rounded-lg')
  })
})
