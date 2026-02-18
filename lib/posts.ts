import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

const postsDirectory = path.join(process.cwd(), 'content/posts')

export interface Post {
  slug: string
  baseSlug?: string
  lang?: string
  title: string
  date: string
  excerpt: string
  tags: string[]
  author: string
  content: string
}

export function getAllPosts(): Post[] {
  if (!fs.existsSync(postsDirectory)) {
    return []
  }

  const fileNames = fs.readdirSync(postsDirectory)
  const allPosts = fileNames
    .filter((fileName) => fileName.endsWith('.md'))
    .map((fileName) => {
      const slug = fileName.replace(/\.md$/, '')
      const fullPath = path.join(postsDirectory, fileName)
      if (!fs.existsSync(fullPath)) {
        return null
      }
      const fileContents = fs.readFileSync(fullPath, 'utf8')
      const { data, content } = matter(fileContents)

      // derive language from filename suffix (e.g. "post-slug.en")
      const langRe = /\.([a-z]{2})$/i
      const langMatch = langRe.exec(String(slug))
      const lang = langMatch ? langMatch[1].toLowerCase() : 'en'
      const baseSlug = String(slug).replace(/\.[a-z]{2}$/i, '')

      // normalize date to YYYY-MM-DD when possible; fall back to original or empty
      let normalizedDate = ''
      if (typeof data.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(data.date)) {
        normalizedDate = data.date
      } else if (data.date) {
        const parsed = new Date(data.date)
        if (Number.isFinite(parsed.getTime())) {
          normalizedDate = parsed.toISOString().slice(0, 10)
        }
      }

      return {
        // keep original slug (filename without .md) for backward compatibility
        slug,
        // additional, safer fields for consumers
        baseSlug,
        lang,
        title: data.title,
        date: normalizedDate || data.date || '',
        excerpt: data.excerpt,
        tags: data.tags || [],
        author: data.author || 'Gustavo Tsuji',
        content,
      }
    })
    .filter(Boolean)

  // Sort posts by date (newest first)
  return allPosts.sort((a, b) => {
    if (a.date < b.date) {
      return 1
    } else {
      return -1
    }
  })
}

export function getPostBySlug(slug: string): Post | null {
  try {
    const fullPath = path.join(postsDirectory, `${slug}.md`)
    const fileContents = fs.readFileSync(fullPath, 'utf8')
    const { data, content } = matter(fileContents)

    // derive language and baseSlug for the provided slug
    const langRe = /\.([a-z]{2})$/i
    const langMatch = langRe.exec(String(slug))
    const lang = langMatch ? langMatch[1].toLowerCase() : 'en'
    const baseSlug = String(slug).replace(/\.[a-z]{2}$/i, '')

    // normalize date similar to getAllPosts
    let normalizedDate = ''
    if (typeof data.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(data.date)) {
      normalizedDate = data.date
    } else if (data.date) {
      const parsed = new Date(data.date)
      if (Number.isFinite(parsed.getTime())) {
        normalizedDate = parsed.toISOString().slice(0, 10)
      }
    }

    return {
      slug,
      baseSlug,
      lang,
      title: data.title,
      date: normalizedDate || data.date || '',
      excerpt: data.excerpt,
      tags: data.tags || [],
      author: data.author || 'Gustavo Tsuji',
      content,
    }
  } catch {
    return null
  }
}

export function getAllTags(): string[] {
  const posts = getAllPosts()
  const tags = new Set<string>()

  posts.forEach((post) => {
    post.tags.forEach((tag) => tags.add(tag))
  })

  // Use localeCompare to ensure consistent, locale-aware alphabetical sorting
  return Array.from(tags).sort((a, b) => a.localeCompare(b))
}
