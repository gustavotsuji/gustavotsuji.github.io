import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'
// Gera todos os caminhos possíveis para exportação estática
export async function generateStaticParams() {
  const postsDir = path.join(process.cwd(), 'content', 'posts')
  const files = fs.readdirSync(postsDir)
  return files
    .filter((file) => /\.[a-z]{2}\.md$/.test(file))
    .map((file) => {
      const slugMatch = file.match(/^(.*)\.([a-z]{2})\.md$/)
      if (!slugMatch) return null
      const slug = slugMatch[1]
      const lang = slugMatch[2]
      const fileContent = fs.readFileSync(path.join(postsDir, file), 'utf8')
      const { data } = matter(fileContent)
      if (!data.date) return null
      const [year, month] = data.date.split('-')
      return { lang, year, month, slug }
    })
    .filter(Boolean)
}
import { notFound } from 'next/navigation'
import React from 'react'
import ReactMarkdown from 'react-markdown'

interface BlogPostPageProps {
  params: {
    lang: string
    year: string
    month: string
    slug: string
  }
}

export default async function BlogPostPage({ params }: BlogPostPageProps) {
  const { lang, year, month, slug } = await params
  // Exemplo de caminho: content/posts/postgresql-partitioning.en.md
  const fileName = `${slug}.${lang}.md`
  const filePath = path.join(process.cwd(), 'content', 'posts', fileName)

  if (!fs.existsSync(filePath)) {
    notFound()
  }

  const fileContent = fs.readFileSync(filePath, 'utf8')
  const { data, content } = matter(fileContent)

  // Valida se a data do frontmatter bate com a URL
  const postDate = new Date(data.date)
  const urlYear = String(postDate.getFullYear())
  const urlMonth = String(postDate.getMonth() + 1).padStart(2, '0')
  if (urlYear !== year || urlMonth !== month) {
    notFound()
  }

  return (
    <article className="prose mx-auto">
      <h1>{data.title}</h1>
      <p className="text-sm text-gray-500 mb-4">
        {data.date} • {data.author}
      </p>
      <ReactMarkdown>{content}</ReactMarkdown>
    </article>
  )
}
