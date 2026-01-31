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
import MarkdownImage from '@/components/MarkdownImage'
import type { Components } from 'react-markdown'

type Params = {
  lang: string
  year: string
  month: string
  slug: string
}

// Helper made exportable for unit testing: resolve featured image path
export function resolveFeaturedImage(imgPath: string | undefined, baseUrl?: string) {
  if (!imgPath) return undefined
  // if already absolute, return as-is
  if (/^https?:\/\//.test(imgPath)) return imgPath

  const basename = path.basename(imgPath)
  const optimizedAvif = `optimized/${basename.replace(path.extname(basename), '.avif')}`
  const optimizedWebp = `optimized/${basename.replace(path.extname(basename), '.webp')}`

  const avifPath = path.join(process.cwd(), 'public', optimizedAvif)
  const webpPath = path.join(process.cwd(), 'public', optimizedWebp)

  const base = baseUrl || process.env.NEXT_PUBLIC_SITE_URL || 'https://gustavotsuji.github.io'

  if (fs.existsSync(avifPath)) return `${base}/${optimizedAvif}`
  if (fs.existsSync(webpPath)) return `${base}/${optimizedWebp}`

  // fallback to original image (assume it's served under site root)
  return `${base}${imgPath}`
}

export default async function BlogPostPage(props: { params: Params | Promise<Params> }) {
  // Next App Router may provide `params` as a Promise in some runtime modes;
  // unwrap it before accessing properties as the framework requires.
  const { lang, year, month, slug } = await props.params
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

  // metadata for head (App Router supports generateMetadata as a separate export,
  // but here we return per-page metadata via in-component Metadata API by constructing head tags)
  const baseUrl = process.env.NEXT_PUBLIC_SITE_URL || 'https://gustavotsuji.github.io'
  const postUrl = `${baseUrl}/blog/${lang}/${year}/${month}/${slug}`
  // Prefer optimized versions (avif/webp) under public/optimized when available
  const resolveFeaturedImage = (imgPath: string | undefined) => {
    if (!imgPath) return undefined
    // if already absolute, return as-is
    if (/^https?:\/\//.test(imgPath)) return imgPath

    const basename = path.basename(imgPath)
    const optimizedAvif = `optimized/${basename.replace(path.extname(basename), '.avif')}`
    const optimizedWebp = `optimized/${basename.replace(path.extname(basename), '.webp')}`

    const avifPath = path.join(process.cwd(), 'public', optimizedAvif)
    const webpPath = path.join(process.cwd(), 'public', optimizedWebp)

    if (fs.existsSync(avifPath)) return `${baseUrl}/${optimizedAvif}`
    if (fs.existsSync(webpPath)) return `${baseUrl}/${optimizedWebp}`

    // fallback to original image (assume it's served under site root)
    return `${baseUrl}${imgPath}`
  }

  const featuredImageUrl = resolveFeaturedImage(data.image)

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: data.title,
    description: data.excerpt || data.description || '',
    author: { '@type': 'Person', name: data.author || 'Gustavo Tsuji' },
    datePublished: data.date,
    mainEntityOfPage: { '@type': 'WebPage', '@id': postUrl },
    image: featuredImageUrl || undefined,
  }
  // determine text direction from language (defaults to ltr)
  const dir = ['ar', 'he', 'fa', 'ur'].includes(lang) ? 'rtl' : 'ltr'

  return (
    <>
      {/* Head meta */}
      <head>
        <title>{data.title}</title>
        {/* Inform the language of the page for user agents and TTS */}
        <meta httpEquiv="Content-Language" content={lang} />
        <meta name="language" content={lang} />
        <meta name="description" content={data.excerpt || ''} />
        <link rel="canonical" href={postUrl} />
        {/* hreflang alternatives - adjust languages you support */}
        <link rel="alternate" hrefLang={lang} href={postUrl} />
        <link rel="alternate" hrefLang="x-default" href={postUrl} />

        {/* Open Graph */}
        <meta property="og:type" content="article" />
        <meta property="og:title" content={data.title} />
        <meta property="og:description" content={data.excerpt || ''} />
        {featuredImageUrl && <meta property="og:image" content={featuredImageUrl} />}

        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={data.title} />
        <meta name="twitter:description" content={data.excerpt || ''} />
      </head>

      {/* JSON-LD */}
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      {/* Preload featured image for faster LCP when available (prefer optimized output) */}
      {featuredImageUrl && <link rel="preload" as="image" href={featuredImageUrl} />}

      {/* use the site header-aware utility so padding/scroll-margin track the header height */}
      <article lang={lang} dir={dir} className="prose mx-auto with-header">
        <h1>{data.title}</h1>
        <p className="text-sm text-gray-500 mb-4">
          {data.date} • {data.author}
        </p>
        <ReactMarkdown components={{ img: MarkdownImage as unknown as Components['img'] }}>
          {content}
        </ReactMarkdown>
      </article>
    </>
  )
}
