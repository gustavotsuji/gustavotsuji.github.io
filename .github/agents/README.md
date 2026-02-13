# GitHub Copilot Agents - Gustavo's Portfolio/Blog

This directory contains specialized AI agents for the Next.js blog/portfolio project. Each agent has deep expertise in a specific domain.

## Available Agents

### 🎨 Frontend Agent

**File**: `frontend-agent.md`  
**Expertise**: React components, Next.js pages, JSX, component patterns
**Activation**: Mention "component", "page", "react", "jsx", "frontend"

### 🎯 Styling Agent

**File**: `styling-agent.md`  
**Expertise**: Tailwind CSS, responsive design, layouts, CSS
**Activation**: Mention "style", "tailwind", "css", "design", "layout"

### 🌍 Internationalization Agent

**File**: `i18n-agent.md`  
**Expertise**: Multi-language support, translations, content management (EN, PT, JA)
**Activation**: Mention "i18n", "translate", "language", "português", "日本語"

### ⚡ Performance & SEO Agent

**File**: `performance-seo-agent.md`  
**Expertise**: Web vitals, LHCI, SEO optimization, performance metrics
**Activation**: Mention "performance", "seo", "lhci", "lighthouse", "metrics"

### ✅ Content Quality Agent

**File**: `content-quality-agent.md`  
**Expertise**: Blog post validation, accuracy checking, detecting hallucinations, fact-checking
**Activation**: Mention "review", "check", "validate", "correct", "accurate", "hallucination"

### 🧪 Testing Agent

**File**: `testing-agent.md`  
**Expertise**: Jest, unit tests, React testing, coverage
**Activation**: Mention "test", "jest", "coverage", "mock"

### 🚀 DevOps Agent

**File**: `devops-agent.md`  
**Expertise**: GitHub Actions, CI/CD, deployment, environment config
**Activation**: Mention "deploy", "workflow", "ci/cd", "github actions"

### 📚 Documentation Agent

**File**: `documentation-agent.md`  
**Expertise**: Project documentation, guides, README updates
**Activation**: Mention "docs", "documentation", "guide"

### 📦 Dependencies Agent

**File**: `dependencies-agent.md`  
**Expertise**: Package updates, compatibility, security vulnerabilities
**Activation**: Mention "update deps", "upgrade", "outdated", "npm audit"

## How It Works

GitHub Copilot automatically routes your questions to the most appropriate agent based on context and keywords. You don't need to manually select an agent - just ask your question naturally in the chat!

Each agent is specialized for its domain and understands the project's unique architecture and requirements.

### Examples

- "Add a new blog post" → Frontend Agent
- "Make the hero section responsive" → Styling Agent
- "Add Portuguese translation for the Hero component" → Internationalization Agent
- "Why is the Lighthouse score dropping?" → Performance & SEO Agent
- "Review this blog post for accuracy" → Content Quality Agent
- "Write tests for the BlogPreview component" → Testing Agent
- "Deploy to GitHub Pages" → DevOps Agent
- "Update packages to latest versions" → Dependencies Agent

## Customization

Feel free to modify any agent's instructions or add new specialized agents for:

- Additional quality checks
- Performance optimization
- Security validation
- Business logic specific to your content

## Agent Structure

Each agent file contains:

- **Expertise**: What the agent knows
- **Context**: Project-specific details
- **Activation Keywords**: When to use this agent
- **Guidelines**: Best practices and patterns
- **Examples**: Code samples and commands
