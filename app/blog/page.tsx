import Link from 'next/link'
import type { Metadata } from 'next'
import { getAllPosts } from '@/lib/posts'

export const metadata: Metadata = {
  title: 'Blog - Gustavo Tsuji',
  description: 'Technical articles about software engineering, cloud architecture, and leadership',
}

export default function BlogPage() {
  const blogPosts = getAllPosts()
  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 pt-24 pb-20">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="mb-12">
            <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">Blog</h1>
            <p className="text-xl text-gray-600 dark:text-gray-400">
              Thoughts on software engineering, cloud architecture, and technical leadership
            </p>
          </div>

          {/* Blog Posts */}
          <div className="space-y-12">
            {blogPosts.map((post) => (
              <article
                key={post.slug}
                className="border-b border-gray-200 dark:border-gray-800 pb-12 last:border-0"
              >
                <div className="flex flex-wrap gap-2 mb-3">
                  {post.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm font-medium"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-3 hover:text-primary-600 dark:hover:text-primary-400 transition-colors">
                  <Link href={`/blog/${post.slug}`}>{post.title}</Link>
                </h2>

                <div className="flex items-center text-sm text-gray-500 dark:text-gray-500 mb-4 space-x-4">
                  <time dateTime={post.date}>
                    {new Date(post.date).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric',
                    })}
                  </time>
                </div>

                <p className="text-gray-700 dark:text-gray-300 mb-4 text-lg leading-relaxed">
                  {post.excerpt}
                </p>

                <Link
                  href={`/blog/${post.slug}`}
                  className="inline-flex items-center text-primary-600 dark:text-primary-400 hover:underline font-semibold"
                >
                  Read full article
                  <svg
                    className="w-4 h-4 ml-1"
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
              </article>
            ))}
          </div>

          {/* Coming Soon Message */}
          <div className="mt-12 p-8 bg-gray-50 dark:bg-gray-800 rounded-lg text-center">
            <p className="text-gray-600 dark:text-gray-400">
              More articles coming soon! Follow me on{' '}
              <a
                href="https://linkedin.com/in/gustavo-tsuji-7100462b"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-600 dark:text-primary-400 hover:underline font-semibold"
              >
                LinkedIn
              </a>{' '}
              for updates.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
