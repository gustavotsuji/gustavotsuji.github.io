'use client'
import Link from 'next/link'
import React from 'react'
import type { Post } from '@/lib/posts'
import BlogPostCard from './BlogPostCard'
import LanguageSelector from './LanguageSelector'

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
  return (
    <section id="blog" className="py-20 bg-white dark:bg-gray-900">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto">
          <div className="mb-6 -mx-4 px-4 sm:mx-0 sm:px-0">
            <LanguageSelector selectedLang={selectedLang} onSelectLang={setLang} />
          </div>

          <div className="space-y-6 mb-12">
            {posts.map((post) => (
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
