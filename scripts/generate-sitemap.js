const fs = require('node:fs')
const path = require('node:path')

const baseUrl = process.env.NEXT_PUBLIC_SITE_URL || 'https://gustavotsuji.github.io'

function walkPosts() {
  const postsDir = path.join(process.cwd(), 'content', 'posts')
  if (!fs.existsSync(postsDir)) return []
  const files = fs.readdirSync(postsDir).filter((f) => f.endsWith('.md'))
  const urls = files
    .map((file) => {
      const match = file.match(/^(.*)\.([a-z]{2})\.md$/)
      if (!match) return null
      const slug = match[1]
      const lang = match[2]
      const content = fs.readFileSync(path.join(postsDir, file), 'utf8')
      const m = content.match(/date:\s*'([0-9-]+)'/) || content.match(/date:\s*"([0-9-]+)"/)
      const date = m ? m[1] : null
      if (!date) return null
      const [year, month] = date.split('-')
      return `${baseUrl}/blog/${lang}/${year}/${month}/${slug}`
    })
    .filter(Boolean)
  return urls
}

function generate() {
  const urls = [baseUrl + '/', baseUrl + '/blog', baseUrl + '/about', ...walkPosts()]
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls
    .map((u) => `  <url>\n    <loc>${u}</loc>\n  </url>`)
    .join('\n')}\n</urlset>`

  const out = path.join(process.cwd(), 'public', 'sitemap.xml')
  fs.writeFileSync(out, sitemap)
  console.warn('Sitemap written to', out)
}

generate()
