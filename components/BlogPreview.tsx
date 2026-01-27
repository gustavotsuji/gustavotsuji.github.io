/* eslint-disable no-unused-vars */
import Link from 'next/link'

interface BlogPreviewProps {
  readonly posts: Array<{
    readonly slug: string
    readonly title: string
    readonly date: string
    readonly excerpt: string
    readonly tags: readonly string[]
    readonly author: string
    readonly content: string
  }>
  readonly selectedLang: string
  readonly setLang?: (_lang: string) => void
}
export default function BlogPreview({ posts = [], selectedLang, setLang }: BlogPreviewProps) {
  const languages = [
    { code: 'pt', label: 'Português' },
    { code: 'en', label: 'English' },
    { code: 'es', label: 'Español' },
    { code: 'ja', label: '日本語' },
    { code: 'fr', label: 'Français' },
  ]
  return (
    <section id="blog" className="py-20 bg-white dark:bg-gray-900">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-end mb-6">
            {setLang && (
              <div className="flex gap-2">
                {languages.map((l) => (
                  <button
                    key={l.code}
                    aria-pressed={selectedLang === l.code}
                    className={`px-3 py-1 rounded text-sm font-medium border transition-colors ${selectedLang === l.code ? 'bg-primary-600 text-white border-primary-600' : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-700 hover:bg-primary-100 dark:hover:bg-primary-900/30'}`}
                    onClick={() => setLang(l.code)}
                  >
                    {l.label}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Latest Articles
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400">
              Thoughts on software engineering, cloud architecture, and technical leadership
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
            {posts.map((post) => (
              <article
                key={post.slug}
                className="bg-gray-50 dark:bg-gray-800 rounded-lg overflow-hidden hover:shadow-xl transition-shadow"
              >
                <div className="p-6">
                  <div className="flex flex-wrap gap-2 mb-3">
                    {post.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded text-xs font-medium"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>

                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                    {post.date ? (
                      <Link
                        href={`/blog/${post.slug.split('.').pop() || 'en'}/${post.date.slice(0, 4)}/${post.date.slice(5, 7)}/${post.slug.replace(/\.[a-z]{2}$/, '')}`}
                      >
                        {post.title}
                      </Link>
                    ) : (
                      <span>{post.title}</span>
                    )}
                  </h3>

                  <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-3">
                    {post.excerpt}
                  </p>

                  <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-500">
                    <time dateTime={post.date}>
                      {new Date(post.date).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                      })}
                    </time>
                    {post.date ? (
                      <Link
                        href={`/blog/${post.slug.split('.').pop() || 'en'}/${post.date.slice(0, 4)}/${post.date.slice(5, 7)}/${post.slug.replace(/\.[a-z]{2}$/, '')}`}
                        className="text-primary-600 dark:text-primary-400 hover:underline font-medium"
                      >
                        Read more →
                      </Link>
                    ) : null}
                  </div>
                </div>
              </article>
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
