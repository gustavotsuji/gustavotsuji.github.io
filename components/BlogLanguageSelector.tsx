'use client'
import { useState } from 'react'
import BlogPreview from './BlogPreview'

interface BlogLanguageSelectorProps {
  allPosts: Array<{
    slug: string
    title: string
    date: string
    excerpt: string
    tags: string[]
    author: string
    content: string
  }>
}

export default function BlogLanguageSelector({ allPosts }: BlogLanguageSelectorProps) {
  const [lang, setLang] = useState('en')
  const posts = allPosts.filter((post) => post.slug.endsWith(`.${lang}`)).slice(0, 5)
  return <BlogPreview posts={posts} selectedLang={lang} setLang={setLang} />
}
