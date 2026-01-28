import fs from 'node:fs'
import path from 'node:path'
import React from 'react'

// Mock react-markdown to avoid ESM import issues during Jest transform
jest.mock(
  'react-markdown',
  () => (props: { children?: React.ReactNode }) => React.createElement('div', null, props.children)
)

// Provide a mockable notFound so tests can assert calls without Next throwing
let notFoundCalled = false
const notFoundMock = () => {
  notFoundCalled = true
}
jest.mock('next/navigation', () => ({ notFound: () => notFoundMock() }))

import BlogPostPage from '../[slug]/page'

type Params = { lang: string; year: string; month: string; slug: string }

describe('BlogPostPage (unit)', () => {
  const postsDir = path.join(process.cwd(), 'content', 'posts')
  beforeAll(() => {
    if (!fs.existsSync(postsDir)) fs.mkdirSync(postsDir, { recursive: true })
  })

  afterAll(() => {
    // leave directory in place but don't remove to avoid accidental deletes
  })

  beforeEach(() => {
    notFoundCalled = false
  })

  it('resolves and does not call notFound when the file exists and date matches params', async () => {
    const props = { params: { lang: 'en', year: '2026', month: '01', slug: 'unit-test-post' } }
    const filePath = path.join(postsDir, 'unit-test-post.en.md')
    const fm = [
      '---',
      'title: "Unit Test Post"',
      'date: "2026-01-15"',
      'excerpt: "x"',
      '---',
      '',
      'body',
    ].join('\n')
    fs.writeFileSync(filePath, fm)
    try {
      await expect(BlogPostPage(props as unknown as { params: Params })).resolves.not.toThrow()
      expect(notFoundCalled).toBe(false)
    } finally {
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath)
    }
  })

  it('calls notFound when the frontmatter date does not match params', async () => {
    const props = { params: { lang: 'en', year: '2025', month: '12', slug: 'unit-test-bad-date' } }
    const filePath = path.join(postsDir, 'unit-test-bad-date.en.md')
    const fm = [
      '---',
      'title: "Bad Date Post"',
      'date: "2026-01-15"',
      'excerpt: "x"',
      '---',
      '',
      'body',
    ].join('\n')
    fs.writeFileSync(filePath, fm)
    try {
      // because we mock notFound to only set a flag (and not throw), the
      // component will still return React elements — assert it resolves
      await expect(BlogPostPage(props as unknown as { params: Params })).resolves.toBeDefined()
      expect(notFoundCalled).toBe(true)
    } finally {
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath)
    }
  })
})
