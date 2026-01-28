/** @type {import('next').NextConfig} */
const baseConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  basePath: '',
  trailingSlash: true,
  // Provide an empty turbopack config so Next.js knows we are aware of Turbopack
  // and suppresses the error when using webpack-based plugins (bundle analyzer).
  turbopack: {},
}

// Allow bundle analyzer when ANALYZE env var is set
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
})

module.exports = withBundleAnalyzer(baseConfig)
