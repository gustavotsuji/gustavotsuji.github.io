# Gustavo Tsuji - Portfolio & Blog

Modern portfolio website built with Next.js, TypeScript, and Tailwind CSS. Features a Medium-style blog for technical articles.

## 🚀 Features

- **Modern Design**: Clean, professional interface with dark mode support
- **Responsive**: Mobile-first design that works on all devices
- **Blog System**: Medium-style blog for technical articles
- **SEO Optimized**: Meta tags and semantic HTML for better search visibility
- **Performance**: Optimized for speed with Next.js static export
- **GitHub Pages Ready**: Configured for easy deployment to GitHub Pages

## 🛠️ Tech Stack

- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Runtime**: Node.js 24 (LTS)
- **Package Manager**: Yarn
- **Testing**: Jest + React Testing Library
- **Code Quality**: ESLint + Prettier + Husky
- **Deployment**: GitHub Pages

## 📋 Prerequisites

- **Node.js**: >= 24.0.0 (LTS)
- **Yarn**: >= 1.22.0 (recommended) or npm >= 10.0.0

Check your versions:

```bash
node --version  # Should be v24.x.x
yarn --version  # Should be >= 1.22.x
```

If you need to upgrade Node.js, see [NODE_UPGRADE_GUIDE.md](./docs/NODE_UPGRADE_GUIDE.md)

To install Yarn:

```bash
npm install -g yarn
```

## 📦 Installation

```bash
yarn install
```

## 🏃‍♂️ Development

```bash
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🏗️ Build

```bash
yarn build
```

This creates an optimized production build in the `out` directory.

## 🧪 Testing

```bash
# Run all tests
yarn test

# Watch mode
yarn test:watch

# Coverage report
yarn test:coverage
```

## ✅ Code Quality

```bash
# Lint code
yarn lint

# Fix lint issues
yarn lint:fix

# Format code
yarn format

# Check formatting
yarn format:check
```

The project uses:

- **ESLint**: Identifies and fixes code issues
- **Prettier**: Enforces consistent code formatting
- **Husky**: Git hooks for pre-commit validation
- **Commitlint**: Validates commit messages (Conventional Commits)

See [CODE_QUALITY_GUIDE.md](./docs/CODE_QUALITY_GUIDE.md) for details.

## 🚀 Deployment to GitHub Pages

### Option 1: Using Yarn Script

```bash
yarn deploy
```

This will build and deploy automatically to GitHub Pages.

### Option 2: Manual Deployment

1. Build the project:

```bash
yarn build
```

2. The static files will be in the `out` directory
3. Push to GitHub and enable GitHub Pages from the `gh-pages` branch

## 📝 Adding Blog Posts

See [BLOG_GUIDE.md](./docs/BLOG_GUIDE.md) for detailed instructions on creating blog posts.

Quick start:

1. Create a new `.md` file in `content/posts/`
2. Add frontmatter (title, date, excerpt, tags, author)
3. Write your content in Markdown
4. The post will appear automatically!

## 📚 Documentation

For detailed technical documentation, see the [docs](./docs) directory:

- **[docs/README.md](./docs/README.md)** - Documentation index
- **[docs/DEPLOY.md](./docs/DEPLOY.md)** - Complete deployment guide
- **[docs/BLOG_GUIDE.md](./docs/BLOG_GUIDE.md)** - How to create blog posts
- **[docs/CODE_QUALITY_GUIDE.md](./docs/CODE_QUALITY_GUIDE.md)** - ESLint, Prettier, and Husky
- **[docs/TESTING_GUIDE.md](./docs/TESTING_GUIDE.md)** - Unit testing guide
- **[docs/NODE_UPGRADE_GUIDE.md](./docs/NODE_UPGRADE_GUIDE.md)** - Node.js 24 upgrade
- **[docs/YARN_MIGRATION.md](./docs/YARN_MIGRATION.md)** - npm to Yarn migration
- **[docs/TAILWIND_V4_GUIDE.md](./docs/TAILWIND_V4_GUIDE.md)** - Tailwind CSS 4 guide
- **[docs/DEPENDABOT_GUIDE.md](./docs/DEPENDABOT_GUIDE.md)** - Automated updates
- [.github/workflows/deploy.yml](./.github/workflows/deploy.yml) - CI/CD configuration

## �📄 License

© 2026 Gustavo Tsuji. All rights reserved.

## 📧 Contact

- Email: gustavokt@gmail.com
- LinkedIn: [gustavo-tsuji-7100462b](https://linkedin.com/in/gustavo-tsuji-7100462b)
- GitHub: [gustavotsuji](https://github.com/gustavotsuji)
