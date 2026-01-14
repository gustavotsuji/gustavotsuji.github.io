# Migração de npm para Yarn

## 📋 O que mudou?

Este projeto agora usa **Yarn** como gerenciador de pacotes ao invés do npm.

## ✅ Por que Yarn?

- ⚡ **Mais rápido**: Cache offline e instalações paralelas
- 🔒 **Mais seguro**: Lockfile consistente e verificação de checksums
- 📦 **Melhor gerenciamento**: Workspaces nativos e deduplicação
- 🎯 **Determinístico**: Instalações idênticas em todos os ambientes
- 💾 **Economia de espaço**: Cache global compartilhado

## 🚀 Comandos Equivalentes

| npm                                | Yarn                        |
| ---------------------------------- | --------------------------- |
| `npm install`                      | `yarn install` ou `yarn`    |
| `npm install [package]`            | `yarn add [package]`        |
| `npm install [package] --save-dev` | `yarn add [package] --dev`  |
| `npm install [package] --global`   | `yarn global add [package]` |
| `npm uninstall [package]`          | `yarn remove [package]`     |
| `npm run [script]`                 | `yarn [script]`             |
| `npm run dev`                      | `yarn dev`                  |
| `npm run build`                    | `yarn build`                |
| `npm run start`                    | `yarn start`                |
| `npm run lint`                     | `yarn lint`                 |
| `npm test`                         | `yarn test`                 |
| `npm outdated`                     | `yarn outdated`             |
| `npm audit`                        | `yarn audit`                |
| `npm cache clean`                  | `yarn cache clean`          |

## 📦 Instalação do Yarn

Se você ainda não tem o Yarn instalado:

```bash
# Via npm (recomendado)
npm install -g yarn

# Via Homebrew (macOS)
brew install yarn

# Via script oficial
curl -o- -L https://yarnpkg.com/install.sh | bash
```

Verificar instalação:

```bash
yarn --version  # Deve mostrar >= 1.22.0
```

## 🔄 Migração do Projeto

### Passo 1: Remover arquivos npm

```bash
# Remover node_modules e package-lock.json
rm -rf node_modules package-lock.json
```

### Passo 2: Instalar com Yarn

```bash
# Instalar dependências
yarn install
```

Isso criará o arquivo `yarn.lock` automaticamente.

### Passo 3: Testar o projeto

```bash
# Testar desenvolvimento
yarn dev

# Testar build
yarn build

# Testar lint
yarn lint
```

## 📝 Scripts Disponíveis

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "export": "next build",
    "deploy": "yarn build && gh-pages -d out"
  }
}
```

### Uso:

```bash
# Desenvolvimento
yarn dev

# Build de produção
yarn build

# Deploy para GitHub Pages
yarn deploy

# Linting
yarn lint
```

## 🔍 Comandos Úteis do Yarn

### Gerenciar dependências

```bash
# Adicionar dependência
yarn add react react-dom

# Adicionar dev dependency
yarn add -D typescript @types/node

# Atualizar dependência
yarn upgrade react

# Remover dependência
yarn remove lodash
```

### Manutenção

```bash
# Ver dependências desatualizadas
yarn outdated

# Atualizar interativamente
yarn upgrade-interactive

# Limpar cache
yarn cache clean

# Verificar integridade
yarn check

# Auditar segurança
yarn audit

# Corrigir vulnerabilidades
yarn audit --fix
```

### Informações

```bash
# Ver informações de um pacote
yarn info [package]

# Listar todas as dependências
yarn list

# Ver por que um pacote está instalado
yarn why [package]
```

## 🔒 yarn.lock

O arquivo `yarn.lock`:

- ✅ **SEMPRE** fazer commit no Git
- ✅ Garante instalações idênticas
- ✅ Lockfile mais confiável que package-lock.json
- ❌ **NUNCA** editar manualmente

## 🎯 CI/CD (GitHub Actions)

O workflow `.github/workflows/deploy.yml` foi atualizado:

```yaml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '24'
    cache: 'yarn' # ← Agora usa cache do Yarn

- name: Install dependencies
  run: yarn install --frozen-lockfile # ← Usa Yarn
```

## 🐛 Troubleshooting

### "command not found: yarn"

```bash
# Instalar Yarn globalmente
npm install -g yarn
```

### Cache corrompido

```bash
# Limpar cache e reinstalar
yarn cache clean
rm -rf node_modules yarn.lock
yarn install
```

### Conflito de dependências

```bash
# Forçar resolução
yarn install --force

# Ou adicionar resolução no package.json
{
  "resolutions": {
    "package-name": "version"
  }
}
```

### Build falha

```bash
# Limpar tudo e reinstalar
rm -rf node_modules yarn.lock .next
yarn install
yarn build
```

## 📚 Recursos

- [Documentação oficial do Yarn](https://classic.yarnpkg.com/en/docs)
- [Migração do npm para Yarn](https://classic.yarnpkg.com/en/docs/migrating-from-npm)
- [Yarn Cheat Sheet](https://devhints.io/yarn)

## ✨ Dicas

1. **Use `yarn` ao invés de `yarn install`** - é mais curto!
2. **Commit o yarn.lock** - sempre!
3. **Use `--frozen-lockfile` no CI** - evita surpresas
4. **Aproveite o cache offline** - trabalhe sem internet
5. **Use `yarn why`** - entenda suas dependências

---

**Nota**: Se você preferir continuar usando npm, tudo bem! O projeto funciona com ambos, mas Yarn é recomendado para melhor performance e consistência.
