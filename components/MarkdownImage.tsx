import React from 'react'
import fs from 'node:fs'
import path from 'node:path'

type ImgProps = Readonly<React.ImgHTMLAttributes<HTMLImageElement>>

/**
 * MarkdownImage renders a <picture> tag preferring optimized assets under
 * /optimized/<original-relative-path> if those files exist (avif, webp),
 * otherwise falls back to the provided src.
 */
export default function MarkdownImage(props: ImgProps) {
  const { src = '', alt = '', width, height } = props

  // Ensure src is a string before doing URL checks (ImgProps allows Blob)
  const srcStr = typeof src === 'string' ? src : ''
  // If it's an absolute URL (http/https), render a plain <img> and avoid
  // server-side fs checks.
  if (srcStr.startsWith('http://') || srcStr.startsWith('https://')) {
    return (
      <img
        src={srcStr}
        alt={alt}
        loading="lazy"
        decoding="async"
        width={width}
        height={height}
        style={{ maxWidth: '100%', height: 'auto' }}
      />
    )
  }

  // Map /images/... -> public/optimized/images/... to check for optimized files.
  const publicDir = path.join(process.cwd(), 'public')
  const normalized = srcStr.replace(/^\//, '') // remove leading slash
  const optimizedAvif = path.join(
    publicDir,
    'optimized',
    normalized.replace(path.extname(normalized), '.avif')
  )
  const optimizedWebp = path.join(
    publicDir,
    'optimized',
    normalized.replace(path.extname(normalized), '.webp')
  )

  const hasAvif = fs.existsSync(optimizedAvif)
  const hasWebp = fs.existsSync(optimizedWebp)

  const optimizedBase = '/optimized/' + normalized.replace(path.extname(normalized), '')

  return (
    <picture>
      {hasAvif && <source srcSet={`${optimizedBase}.avif`} type="image/avif" />}
      {hasWebp && <source srcSet={`${optimizedBase}.webp`} type="image/webp" />}
      <img
        src={src}
        alt={alt}
        loading="lazy"
        decoding="async"
        width={width}
        height={height}
        style={{ maxWidth: '100%', height: 'auto' }}
      />
    </picture>
  )
}
