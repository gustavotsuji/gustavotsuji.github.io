'use client'
import Link from 'next/link'
import React from 'react'
import type { Post } from '@/lib/posts'

interface BlogPreviewClientProps {
  readonly posts: readonly Post[]
  readonly selectedLang?: string
  readonly setLang?: React.Dispatch<React.SetStateAction<string>>
}

export default function BlogPreviewClient({
  posts = [],
  selectedLang = 'en',
  setLang,
}: BlogPreviewClientProps) {
  const languages = [
    { code: 'pt', label: 'Português', short: 'PT' },
    { code: 'en', label: 'English', short: 'EN' },
    { code: 'es', label: 'Español', short: 'ES' },
    { code: 'ja', label: '日本語', short: 'JA' },
    { code: 'fr', label: 'Français', short: 'FR' },
  ]

  return (
    <section id="blog" className="py-20 bg-white dark:bg-gray-900">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <div className="mb-6 -mx-4 px-4 sm:mx-0 sm:px-0">
            {setLang && (
              <div className="overflow-x-auto pb-2 scrollbar-hide">
                <div className="flex gap-2 justify-start sm:justify-end min-w-max sm:min-w-0">
                  {languages.map((l) => (
                    <button
                      key={l.code}
                      aria-pressed={selectedLang === l.code}
                      aria-label={l.label}
                      className={`px-3 py-2 rounded text-sm font-medium border transition-colors whitespace-nowrap flex-shrink-0 ${selectedLang === l.code ? 'bg-primary-600 text-white border-primary-600' : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-700 hover:bg-primary-100 dark:hover:bg-primary-900/30'}`}
                      onClick={() => setLang?.(l.code)}
                    >
                      <span className="hidden sm:inline">{l.label}</span>
                      <span className="sm:hidden">{l.short}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="space-y-6 mb-12">
            {posts.map((post) => {
              const dateIso =
                typeof post.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(post.date)
                  ? post.date
                  : ''
              const year = dateIso ? dateIso.slice(0, 4) : '0000'
              const month = dateIso ? dateIso.slice(5, 7) : '01'

              const langFromPost = (post.lang as string) || ''
              const baseSlugFromPost = (post.baseSlug as string) || ''

              const rawLang = langFromPost || String(post.slug).split('.').pop() || 'en'
              const langCode = /^[a-z]{2}$/.test(rawLang) ? rawLang : 'en'
              const rawSlug = baseSlugFromPost || String(post.slug).replace(/\.[a-z]{2}$/, '')

              const safeLang = encodeURIComponent(langCode)
              const safeSlug = encodeURIComponent(rawSlug)

              const safeHref = `/blog/${safeLang}/${year}/${month}/${safeSlug}`

              return (
                <article
                  key={post.slug}
                  className="bg-gray-50 dark:bg-gray-800 rounded-lg overflow-hidden hover:shadow-xl transition-all hover:-translate-y-1 border border-transparent hover:border-primary-200 dark:hover:border-primary-800"
                >
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
                        {new Date(post.date).toLocaleDateString('en-US', {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric',
                        })}
                      </time>
                    </div>

                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-3 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                      {post.date ? (
                        <Link href={safeHref}>{post.title}</Link>
                      ) : (
                        <span>{post.title}</span>
                      )}
                    </h3>

                    <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-2 text-base">
                      {post.excerpt}
                    </p>

                    <div className="flex items-center justify-end">
                      {post.date ? (
                        <Link
                          href={safeHref}
                          className="text-primary-600 dark:text-primary-400 hover:underline font-medium inline-flex items-center gap-1"
                        >
                          Read more
                          <svg
                            className="w-4 h-4"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
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
            })}
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
