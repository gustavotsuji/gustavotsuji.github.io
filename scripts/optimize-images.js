#!/usr/bin/env node
/**
 * scripts/optimize-images.js
 * - Scans `public/images` for jpg/png files (recursively) and writes optimized
 *   AVIF and WebP variants under `public/optimized/<relative-path>`.
 * - Uses `sharp` when available. If not installed, the script exits with instructions.
 */
const fs = require('fs')
const path = require('path')

async function main() {
  let sharp
  try {
    sharp = require('sharp')
  } catch {
    console.error('`sharp` is not installed. Install with: yarn add --dev sharp')
    process.exit(1)
  }

  const publicDir = path.join(process.cwd(), 'public')
  const imagesDir = path.join(publicDir, 'images')
  const outDir = path.join(publicDir, 'optimized')

  if (!fs.existsSync(imagesDir)) {
    console.error('No public/images directory found — nothing to optimize.')
    process.exit(0)
  }

  const exts = ['.jpg', '.jpeg', '.png']

  function walk(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true })
    let files = []
    for (const e of entries) {
      const p = path.join(dir, e.name)
      if (e.isDirectory()) files = files.concat(walk(p))
      else files.push(p)
    }
    return files
  }

  const files = walk(imagesDir).filter((f) => exts.includes(path.extname(f).toLowerCase()))
  if (!files.length) {
    console.warn('No JPG/PNG images found under public/images')
    return
  }

  for (const file of files) {
    const rel = path.relative(imagesDir, file)
    const destDir = path.join(outDir, path.dirname(rel))
    if (!fs.existsSync(destDir)) fs.mkdirSync(destDir, { recursive: true })

    const baseName = path.basename(rel, path.extname(rel))
    const outAvif = path.join(destDir, baseName + '.avif')
    const outWebp = path.join(destDir, baseName + '.webp')

    try {
      await sharp(file).avif({ quality: 50 }).toFile(outAvif)
      await sharp(file).webp({ quality: 60 }).toFile(outWebp)
      console.warn('optimized', rel)
    } catch (err) {
      console.error('failed to optimize', rel, err?.message || String(err))
    }
  }

  console.warn('Optimized images written to public/optimized')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
