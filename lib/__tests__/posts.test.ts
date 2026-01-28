import fs from 'node:fs'
import path from 'node:path'
import { getAllPosts, getPostBySlug } from '@/lib/posts'
import type { Post } from '@/lib/posts'

// We'll create temporary markdown files in the content/posts folder for testing
const postsDir = path.join(process.cwd(), 'content/posts')

function writePostFile(fileName: string, frontmatter: Record<string, unknown>, body = '') {
  const fmLines = ['---']
  for (const [k, v] of Object.entries(frontmatter)) {
    fmLines.push(`${k}: ${JSON.stringify(v)}`)
  }
  fmLines.push('---', '', body)
  fs.writeFileSync(path.join(postsDir, fileName), fmLines.join('\n'))
}

beforeAll(() => {
  if (!fs.existsSync(postsDir)) fs.mkdirSync(postsDir, { recursive: true })
})

afterEach(() => {
  // cleanup test-created files
  const files = fs.readdirSync(postsDir)
  for (const f of files) {
    if (f.startsWith('test-')) fs.unlinkSync(path.join(postsDir, f))
  }
})

afterAll(() => {
  // no-op: keep folder if existed
})

test('getAllPosts returns normalized lang and baseSlug for files with suffix', () => {
  writePostFile('test-hello.en.md', { title: 'Hello', date: '2021-03-05', excerpt: 'x' }, 'content')
  const posts = getAllPosts()
  const p = posts.find((x) => x.slug === 'test-hello.en')
  expect(p).toBeDefined()
  const post = p as Post
  expect(post.lang).toBe('en')
  expect(post.baseSlug).toBe('test-hello')
  expect(post.date).toBe('2021-03-05')
})

test('getAllPosts normalizes parseable date formats to YYYY-MM-DD', () => {
  writePostFile('test-date.en.md', { title: 'DateTest', date: 'March 10, 2022' }, 'body')
  const posts = getAllPosts()
  const p = posts.find((x) => x.slug === 'test-date.en')
  expect(p).toBeDefined()
  const post = p as Post
  // date should be converted to ISO date (YYYY-MM-DD)
  expect(post.date).toMatch(/^\d{4}-\d{2}-\d{2}$/)
})

test('getPostBySlug returns null for non-existing slug and returns normalized fields for existing', () => {
  writePostFile('test-single.pt.md', { title: 'Single', date: '2020-01-02' }, 'b')
  const existing = getPostBySlug('test-single.pt')
  expect(existing).not.toBeNull()
  const e = existing as Post
  expect(e.lang).toBe('pt')
  expect(e.baseSlug).toBe('test-single')

  const missing = getPostBySlug('does-not-exist.en')
  expect(missing).toBeNull()
})
