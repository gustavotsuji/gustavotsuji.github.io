import fs from 'node:fs'
import path from 'node:path'
import React from 'react'

// Mock react-markdown to avoid importing ESM during Jest transform
jest.mock(
  'react-markdown',
  () => (props: { children?: React.ReactNode }) => React.createElement('div', null, props.children)
)

import BlogPostPage from '../page'

// Provide a minimal mock post file and mock getAllPosts to not affect this test
jest.mock('@/lib/posts', () => ({
  getAllPosts: jest.fn(() => []),
}))

type Params = { lang: string; year: string; month: string; slug: string }

describe('BlogPostPage server component', () => {
  it('renders without crashing when given params', async () => {
    const props = { params: { lang: 'en', year: '2026', month: '01', slug: 'test-post' } }
    const postsDir = path.join(process.cwd(), 'content', 'posts')
    if (!fs.existsSync(postsDir)) fs.mkdirSync(postsDir, { recursive: true })
    const filePath = path.join(postsDir, 'test-post.en.md')
    const fm = [
      '---',
      'title: "Test Post"',
      'date: "2026-01-15"',
      'excerpt: "x"',
      '---',
      '',
      'body',
    ].join('\n')
    fs.writeFileSync(filePath, fm)
    try {
      // ensure the function doesn't throw when invoked
      await expect(BlogPostPage(props as unknown as { params: Params })).resolves.not.toThrow()
    } finally {
      // cleanup
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath)
    }
  })
})
