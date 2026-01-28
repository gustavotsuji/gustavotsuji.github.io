'use client'
import { useState } from 'react'
import BlogPreview from './BlogPreview'
import type { Post } from '@/lib/posts'

interface BlogLanguageSelectorProps {
  readonly allPosts: readonly Post[]
}

export default function BlogLanguageSelector({ allPosts }: BlogLanguageSelectorProps) {
  const [lang, setLang] = useState('en')
  const allowed = ['pt', 'en', 'es', 'ja', 'fr']
  const safeLang = allowed.includes(lang) ? lang : 'en'
  // prefer explicit `lang` field from ingestion, fall back to filename suffix
  const posts = allPosts
    .filter(
      (post) =>
        String(post.lang || post.slug).endsWith(`.${safeLang}`) || String(post.lang) === safeLang
    )
    .slice(0, 5)

  return <BlogPreview posts={posts} selectedLang={lang} setLang={setLang} />
}
