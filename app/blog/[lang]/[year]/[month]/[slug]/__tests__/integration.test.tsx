import fs from 'node:fs'
import path from 'node:path'
import React from 'react'

// Mock react-markdown to avoid ESM handling in tests
jest.mock(
  'react-markdown',
  () => (props: { children?: React.ReactNode }) => React.createElement('div', null, props.children)
)

import BlogPostPage, { generateStaticParams } from '../page'

const postsDir = path.join(process.cwd(), 'content', 'posts')

function writePost(fileName: string, fm: Record<string, unknown>, body = '') {
  const lines = ['---']
  for (const [k, v] of Object.entries(fm)) {
    lines.push(`${k}: ${JSON.stringify(v)}`)
  }
  lines.push('---', '', body)
  fs.writeFileSync(path.join(postsDir, fileName), lines.join('\n'))
}

beforeAll(() => {
  if (!fs.existsSync(postsDir)) fs.mkdirSync(postsDir, { recursive: true })
})

afterEach(() => {
  const files = fs.readdirSync(postsDir)
  for (const f of files) {
    if (f.startsWith('int-test-')) fs.unlinkSync(path.join(postsDir, f))
  }
})

type Params = { lang: string; year: string; month: string; slug: string }

describe('Blog post integration', () => {
  it('generateStaticParams discovers posts and BlogPostPage loads file', async () => {
    writePost(
      'int-test-post.en.md',
      { title: 'Integration', date: '2026-01-28', excerpt: 'x' },
      'content body'
    )

    const params = await generateStaticParams()
    // find our slug
    const found = params.find(
      (p: Params | null) => !!p && p.slug === 'int-test-post' && p.lang === 'en'
    )
    expect(found).toBeTruthy()

    // call the server component with params to ensure it can render without throwing
    const props = { params: { lang: 'en', year: '2026', month: '01', slug: 'int-test-post' } }
    await expect(BlogPostPage(props as unknown as { params: Params })).resolves.not.toThrow()
  })
})
