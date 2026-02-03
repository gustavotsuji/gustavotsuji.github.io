---
title: "Stop Breaking the Pipeline: How 'Shift-Left' and Husky Can Save Your Day (and the Company's Budget)"
date: '2026-01-26'
excerpt: 'Bring pipeline checks to the developer machine with Husky — avoid CI breakages, save time and resources. Practical guide with configuration and examples.'
tags: ['Husky', 'Git', 'CI/CD', 'Shift-Left', 'DevOps', 'Quality Assurance']
author: 'Gustavo Tsuji'
---

Who hasn't been there: you finish a feature, open the Pull Request and anxiously wait for the green check from the pipeline. Ten minutes later, the CI fails. Why? A simple lint error or a Sonar validation that could have been fixed in seconds on your machine.

Beyond frustration, this causes many CI runs (GitHub Actions or similar), increasing costs and wasting compute resources. Worse: every trivial fix requires a new push, restarting the pipeline and invalidating approvals already given in the Code Review.

In this article I share an approach to bring these checks into the developer environment (Shift-Left) using Husky, with a tailored configuration strategy that doesn't block team productivity.

## The "Shift-Left" Concept

The core idea is simple: what if we could run all the steps locally to increase the chance of success and avoid wasting time in CI? Shift-Left brings quality checks (Lint, Tests, Sonar, Security) to the developer's machine before the code reaches the server.

## The Guardian: Husky

We orchestrate this with Husky. It manages scripts triggered automatically by Git events (Git Hooks). Husky acts as a gatekeeper: if a check fails, the git command (commit or push) is aborted instantly.

We split the strategy into two main moments:

1. **Pre-commit:** Run formatters and linters only on staged files.

2. **Pre-push:** Run heavier tests and security checks before sending to the remote repository.

## The clever part: Per-developer customisation

One of the main concerns when adopting local hooks is slowness. "Do I have to run Sonar every time I push?" The answer is: it depends on you.

To solve this, we created an untracked local control file. The developer creates a file at the repository root (e.g. `.husky.user.config`) that is ignored by `.gitignore`.

In this file we define boolean variables to enable or disable specific checks:

```bash
# .husky.user.config
# By DEFAULT, steps can be disabled to save time.
# The developer enables what makes sense for their workflow.

export HUSKY_RUN_LINT=true           # ESLint + Prettier
export HUSKY_RUN_GITLEAKS=true       # Secrets detection
export HUSKY_RUN_UNIT_TESTS=true     # Unit tests
export HUSKY_RUN_TRIVY_SCAN=true     # Vulnerability scanner
export HUSKY_RUN_SONAR_SCAN=false    # SonarQube (heavy, enable when needed)

```

Husky scripts are adapted to read this file before executing. If the variable is `false` or missing, the step is skipped.

## What we validate

With this structure we ensure multiple layers of quality before code leaves the machine:

### Pre-commit:

- **Lint & Prettier:** code style enforcement.
- **GitLeaks:** prevents committing API keys or secrets — if found, the commit is aborted.
- **Dockerfile validation:** if Dockerfile changed, attempt a build to ensure the image is not broken.

### Pre-push:

- **Unit tests:** basic correctness — fail fast.
- **Trivy:** dependency and container vulnerability scanning.
- **NPM Audit:** checks for high-severity vulnerabilities in packages.

### SonarQube integration (local)

When enabled (`HUSKY_RUN_SONAR_SCAN=true`) we:

1. Check if a SonarQube container is running via Docker; if not, start it.
2. Verify the project exists in the local Sonar; if not, create it via the API.
3. Run `sonar-scanner` (via Docker).
4. Query the Quality Gate via API. If the status is ERROR the push is aborted and the developer receives the dashboard link to fix issues.

## Conclusion

Although it's possible to skip checks with `--no-verify` (but you shouldn't 😉), the goal of this architecture is to empower the developer.

By running selective validations locally we drastically reduce CI wait time, save money on Actions minutes, and keep the team sane by avoiding trivial pipeline breaks.

What about you — already using Shift-Left in your dev flow?

---

_Article based on a technical talk about Husky and Git Hooks automation._
