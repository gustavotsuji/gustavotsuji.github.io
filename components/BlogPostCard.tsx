import Link from 'next/link'
import type { Post } from '@/lib/posts'

interface BlogPostCardProps {
  readonly post: Post
}

const getPostHref = (post: Post): string | null => {
  const dateIso =
    typeof post.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(post.date) ? post.date : ''
  if (!dateIso) return null

  const year = dateIso.slice(0, 4)
  const month = dateIso.slice(5, 7)

  const langFromPost = (post.lang as string) || ''
  const baseSlugFromPost = (post.baseSlug as string) || ''

  const rawLang = langFromPost || String(post.slug).split('.').pop() || 'en'
  const langCode = /^[a-z]{2}$/.test(rawLang) ? rawLang : 'en'
  const rawSlug = baseSlugFromPost || String(post.slug).replace(/\.[a-z]{2}$/, '')

  const safeLang = encodeURIComponent(langCode)
  const safeSlug = encodeURIComponent(rawSlug)

  return `/blog/${safeLang}/${year}/${month}/${safeSlug}`
}

export default function BlogPostCard({ post }: BlogPostCardProps) {
  const safeHref = getPostHref(post)

  const formattedDate = new Date(post.date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })

  return (
    <article className="bg-gray-50 dark:bg-gray-800 rounded-lg overflow-hidden hover:shadow-xl transition-all hover:-translate-y-1 border border-transparent hover:border-primary-200 dark:hover:border-primary-800">
      <div className="p-6">
        <div className="flex items-start justify-between gap-4 mb-3">
          <div className="flex flex-wrap gap-2 flex-1">
            {post.tags.map((tag) => (
              <span
                key={tag}
                className="px-2 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded text-xs font-medium"
              >
                {tag}
              </span>
            ))}
          </div>
          <time
            dateTime={post.date}
            className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap"
          >
            {formattedDate}
          </time>
        </div>

        <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-3 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
          {safeHref ? <Link href={safeHref}>{post.title}</Link> : <span>{post.title}</span>}
        </h3>

        <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-2 text-base">
          {post.excerpt}
        </p>

        <div className="flex items-center justify-end">
          {safeHref ? (
            <Link
              href={safeHref}
              className="text-primary-600 dark:text-primary-400 hover:underline font-medium inline-flex items-center gap-1"
            >
              Read more
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 5l7 7-7 7"
                />
              </svg>
            </Link>
          ) : null}
        </div>
      </div>
    </article>
  )
}
