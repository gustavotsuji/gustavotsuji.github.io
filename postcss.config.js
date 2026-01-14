/** @type {import('postcss-load-config').Config} */
module.exports = {
  plugins: {
    '@tailwindcss/postcss': {
      plugins: ['@tailwindcss/typography'],
    },
    autoprefixer: {},
  },
}
