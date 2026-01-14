import { render, screen } from '@testing-library/react'
import BlogPost, { generateMetadata, generateStaticParams } from '../page'

// Mock do next/navigation
jest.mock('next/navigation', () => ({
  notFound: jest.fn(),
}))

// Mock do lib/posts
jest.mock('@/lib/posts', () => ({
  getPostBySlug: jest.fn(),
  getAllPosts: jest.fn(),
}))

// Mock do react-markdown
jest.mock('react-markdown', () => {
  return function MockReactMarkdown({ children }: { children: string }) {
    return <div data-testid="markdown-content">{children}</div>
  }
})

// Mock do react-syntax-highlighter
jest.mock('react-syntax-highlighter', () => ({
  Prism: function MockSyntaxHighlighter({ children }: { children: string }) {
    return <pre data-testid="code-block">{children}</pre>
  },
}))

jest.mock('react-syntax-highlighter/dist/esm/styles/prism', () => ({
  vscDarkPlus: {},
}))

const mockPost = {
  slug: 'test-post',
  title: 'Test Post Title',
  date: '2026-01-10',
  excerpt: 'This is a test post excerpt',
  content: '# Test Content\n\nThis is the test post content.',
  tags: ['testing', 'jest'],
  author: 'Gustavo Tsuji',
}

describe('BlogPost Page', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders post when found', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(mockPost)
    const params = Promise.resolve({ slug: 'test-post' })
    const Component = await BlogPost({ params })
    render(Component)
    expect(screen.getByText('Test Post Title')).toBeInTheDocument()
  })

  it('renders post date', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(mockPost)
    const params = Promise.resolve({ slug: 'test-post' })
    const Component = await BlogPost({ params })
    render(Component)
    // Date is formatted in Portuguese by default
    expect(screen.getByText(/janeiro de 2026/i)).toBeInTheDocument()
  })

  it('renders post tags', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(mockPost)
    const params = Promise.resolve({ slug: 'test-post' })
    const Component = await BlogPost({ params })
    render(Component)
    expect(screen.getByText('testing')).toBeInTheDocument()
    expect(screen.getByText('jest')).toBeInTheDocument()
  })

  it('renders markdown content', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(mockPost)
    const params = Promise.resolve({ slug: 'test-post' })
    const Component = await BlogPost({ params })
    render(Component)
    expect(screen.getByTestId('markdown-content')).toBeInTheDocument()
  })

  it('calls notFound when post does not exist', async () => {
    const { notFound } = require('next/navigation')
    notFound.mockImplementation(() => {
      throw new Error('NEXT_NOT_FOUND')
    })
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(null)
    const params = Promise.resolve({ slug: 'non-existent' })

    await expect(BlogPost({ params })).rejects.toThrow('NEXT_NOT_FOUND')
    expect(notFound).toHaveBeenCalled()
  })

  it('renders article semantic tag', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(mockPost)
    const params = Promise.resolve({ slug: 'test-post' })
    const Component = await BlogPost({ params })
    const { container } = render(Component)
    const article = container.querySelector('article')
    expect(article).toBeInTheDocument()
  })

  it('renders author information', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(mockPost)
    const params = Promise.resolve({ slug: 'test-post' })
    const Component = await BlogPost({ params })
    render(Component)
    expect(screen.getByText('Gustavo Tsuji')).toBeInTheDocument()
  })
})

describe('generateStaticParams', () => {
  it('returns array of slugs', async () => {
    const { getAllPosts } = require('@/lib/posts')
    getAllPosts.mockReturnValue([{ slug: 'post-1' }, { slug: 'post-2' }, { slug: 'post-3' }])

    const result = await generateStaticParams()

    expect(result).toEqual([{ slug: 'post-1' }, { slug: 'post-2' }, { slug: 'post-3' }])
  })

  it('calls getAllPosts', async () => {
    const { getAllPosts } = require('@/lib/posts')
    getAllPosts.mockReturnValue([])
    await generateStaticParams()
    expect(getAllPosts).toHaveBeenCalled()
  })

  it('handles empty posts array', async () => {
    const { getAllPosts } = require('@/lib/posts')
    getAllPosts.mockReturnValue([])
    const result = await generateStaticParams()
    expect(result).toEqual([])
  })
})

describe('generateMetadata', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('returns correct metadata when post exists', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(mockPost)
    const params = Promise.resolve({ slug: 'test-post' })

    const metadata = await generateMetadata({ params })

    expect(metadata).toEqual({
      title: 'Test Post Title | Gustavo Tsuji',
      description: 'This is a test post excerpt',
    })
  })

  it('returns not found metadata when post does not exist', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(null)
    const params = Promise.resolve({ slug: 'non-existent' })

    const metadata = await generateMetadata({ params })

    expect(metadata).toEqual({
      title: 'Post Not Found',
    })
  })

  it('calls getPostBySlug with correct slug', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    getPostBySlug.mockReturnValue(mockPost)
    const params = Promise.resolve({ slug: 'test-slug' })

    await generateMetadata({ params })

    expect(getPostBySlug).toHaveBeenCalledWith('test-slug')
  })

  it('handles different post titles', async () => {
    const { getPostBySlug } = require('@/lib/posts')
    const differentPost = {
      ...mockPost,
      title: 'Different Title',
      excerpt: 'Different excerpt',
    }
    getPostBySlug.mockReturnValue(differentPost)
    const params = Promise.resolve({ slug: 'different-post' })

    const metadata = await generateMetadata({ params })

    expect(metadata.title).toBe('Different Title | Gustavo Tsuji')
    expect(metadata.description).toBe('Different excerpt')
  })
})
