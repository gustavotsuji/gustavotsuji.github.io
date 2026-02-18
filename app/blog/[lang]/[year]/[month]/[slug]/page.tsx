import fs from 'node:fs'
import path from 'node:path'
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
      const fullPath = path.join(postsDir, file)
      if (!fs.existsSync(fullPath)) return null
      const fileContent = fs.readFileSync(fullPath, 'utf8')
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
import type { Metadata } from 'next'

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

// Per-page metadata for the App Router (avoids rendering <head> inside the component)
export async function generateMetadata({
  params,
}: {
  params: Params | Promise<Params>
}): Promise<Metadata> {
  try {
    // `params` may be a Promise in some Next.js runtime modes; unwrap it.
    const { lang, year, month, slug } = await params
    const fileName = `${slug}.${lang}.md`
    const filePath = path.join(process.cwd(), 'content', 'posts', fileName)
    if (!fs.existsSync(filePath)) return {}
    const fileContent = fs.readFileSync(filePath, 'utf8')
    const { data } = matter(fileContent)

    const baseUrl = process.env.NEXT_PUBLIC_SITE_URL || 'https://gustavotsuji.github.io'
    const postUrl = `${baseUrl}/blog/${lang}/${year}/${month}/${slug}`
    const featuredImageUrl = resolveFeaturedImage(data.image)

    const title = data.title || ''
    const description = data.excerpt || data.description || ''

    return {
      title,
      description,
      alternates: {
        canonical: postUrl,
        languages: {
          [lang]: postUrl,
        },
      },
      openGraph: {
        title,
        description,
        url: postUrl,
        type: 'article',
        publishedTime: data.date,
        images: featuredImageUrl ? [{ url: featuredImageUrl }] : undefined,
      },
      twitter: {
        card: 'summary_large_image',
        title,
        description,
        images: featuredImageUrl ? [featuredImageUrl] : undefined,
      },
    }
  } catch (e) {
    // Log the error for diagnostics and return empty metadata so the page still renders

    console.error('generateMetadata error:', e)
    return {}
  }
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

  const baseUrl = process.env.NEXT_PUBLIC_SITE_URL || 'https://gustavotsuji.github.io'
  const postUrl = `${baseUrl}/blog/${lang}/${year}/${month}/${slug}`
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
        <ReactMarkdown
          components={{
            img: MarkdownImage as unknown as Components['img'],
            div: DivRenderer,
          }}
        >
          {content}
        </ReactMarkdown>
      </article>
    </>
  )
}

// Top-level renderer moved out of the React component to satisfy lint rules
type DivRendererProps = React.HTMLAttributes<HTMLDivElement> & { node?: unknown }

function DivRenderer(props: DivRendererProps) {
  const { className, children, ...rest } = props
  const classes = (className || '').split(/\s+/)
  if (classes.includes('callout')) {
    // Render as a semantic container when markdown contains callout HTML
    // (we intentionally render a plain div so styling from `.callout` in CSS applies)
    return (
      <div className={className} {...(rest as React.HTMLAttributes<HTMLDivElement>)}>
        {children}
      </div>
    )
  }

  // default div rendering
  return (
    <div className={className} {...(rest as React.HTMLAttributes<HTMLDivElement>)}>
      {children}
    </div>
  )
}
