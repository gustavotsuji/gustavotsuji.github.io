import Link from 'next/link'
import type { Post } from '@/lib/posts'
import { getAllPosts } from '@/lib/posts'
import BlogPostCard from './BlogPostCard'
import LanguageSelector from './LanguageSelector'

interface BlogPreviewProps {
  readonly posts?: readonly Post[]
  readonly selectedLang?: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly setLang?: any
  readonly showTitle?: boolean
}

export default function BlogPreview({
  posts,
  selectedLang,
  setLang,
  showTitle = true,
}: BlogPreviewProps) {
  // if posts not provided (used on the homepage), load them server-side
  let blogPosts: readonly Post[] = posts ?? []
  if ((!posts || posts.length === 0) && typeof getAllPosts === 'function') {
    // safe to call on server components - filter for English only on homepage
    blogPosts = getAllPosts()
      .filter((post) => {
        const langFromPost = (post.lang as string) || ''
        const langCode = langFromPost || String(post.slug).split('.').pop() || 'en'
        return langCode === 'en'
      })
      .slice(0, 5)
  }

  return (
    <section id="blog" className="py-20 bg-white dark:bg-gray-900">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <div className="mb-6 -mx-4 px-4 sm:mx-0 sm:px-0">
            <LanguageSelector selectedLang={selectedLang} onSelectLang={setLang} />
          </div>

          {showTitle && (
            <div className="text-center mb-12">
              <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-4">
                Latest Articles
              </h2>
              <p className="text-base sm:text-lg text-gray-600 dark:text-gray-400">
                Thoughts on software engineering, cloud architecture, and technical leadership
              </p>
            </div>
          )}

          <div className="space-y-6 mb-12">
            {blogPosts.map((post) => (
              <BlogPostCard key={post.slug} post={post} />
            ))}
          </div>

          <div className="text-center">
            <Link
              href="/blog"
              className="inline-block px-8 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold transition-colors"
            >
              View All Articles
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}
