# Code Quality Guide

Este guia explica como usar as ferramentas de qualidade de código configuradas no projeto: ESLint, Prettier e Husky.

## 📋 Ferramentas

### ESLint

Linter JavaScript/TypeScript que identifica e corrige problemas no código.

**Configuração:** `eslint.config.mjs`

**Regras principais:**

- `@typescript-eslint/no-unused-vars`: Avisa sobre variáveis não utilizadas
- `@typescript-eslint/no-explicit-any`: Avisa sobre uso de `any`
- `no-console`: Avisa sobre uso de `console.log` (permite `console.warn` e `console.error`)
- `prefer-const`: Força uso de `const` quando variável não é reatribuída
- `no-var`: Proíbe uso de `var`

### Prettier

Formatador de código automático que garante estilo consistente.

**Configuração:** `.prettierrc`

**Regras:**

- Sem ponto e vírgula (`semi: false`)
- Aspas simples (`singleQuote: true`)
- Largura máxima de 100 caracteres (`printWidth: 100`)
- 2 espaços de indentação (`tabWidth: 2`)

### Husky

Gerenciador de Git hooks que executa ações antes de commits e pushes.

**Configuração:** `.husky/` directory

**Hooks configurados:**

- **pre-commit**: Executa lint-staged e testes antes de cada commit
- **commit-msg**: Valida mensagem de commit usando commitlint

### Commitlint

Valida mensagens de commit seguindo o padrão Conventional Commits.

**Configuração:** `commitlint.config.js`

**Formato:** `<tipo>(<escopo>): <descrição>`

**Tipos permitidos:**

- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Mudanças na documentação
- `style`: Mudanças de formatação (não afetam o código)
- `refactor`: Refatoração de código
- `perf`: Melhorias de performance
- `test`: Adição ou correção de testes
- `chore`: Mudanças em ferramentas, configurações, dependências
- `ci`: Mudanças em arquivos de CI
- `build`: Mudanças no sistema de build
- `revert`: Reversão de commit anterior

**Exemplos válidos:**

```bash
feat: adiciona autenticação OAuth
fix: corrige erro no cálculo de preço
docs: atualiza README com instruções
refactor: simplifica lógica do componente Header
test: adiciona testes para BlogPreview
chore: atualiza dependências
```

## 🚀 Comandos

### Verificar código

```bash
# Executar ESLint
npx eslint .

# Verificar formatação com Prettier
yarn format:check
```

### Corrigir código

```bash
# Corrigir problemas do ESLint automaticamente
yarn lint:fix

# Formatar código com Prettier
yarn format
```

### Executar testes

```bash
# Executar todos os testes
yarn test

# Executar testes em modo watch
yarn test:watch

# Gerar relatório de cobertura
yarn test:coverage
```

## 🔄 Workflow

### 1. Durante o desenvolvimento

Você pode executar os comandos manualmente:

```bash
# Verificar código
yarn lint:fix
yarn format

# Rodar testes
yarn test
```

### 2. Antes do commit

O Husky executará automaticamente:

1. **lint-staged**: Executa ESLint e Prettier apenas nos arquivos modificados
2. **Testes**: Executa todos os testes

Se houver erros, o commit será bloqueado até que sejam corrigidos.

### 3. Ao fazer commit

O commitlint validará a mensagem do commit:

```bash
# ✅ Válido
git commit -m "feat: adiciona página de contato"

# ❌ Inválido (sem tipo)
git commit -m "adiciona página de contato"

# ❌ Inválido (tipo incorreto)
git commit -m "add: adiciona página de contato"
```

## 📝 lint-staged

O lint-staged executa comandos apenas nos arquivos que estão no stage do Git.

**Configuração em `package.json`:**

```json
"lint-staged": {
  "*.{js,jsx,ts,tsx}": [
    "eslint --fix",
    "prettier --write"
  ]
}
```

Isso significa que:

- Arquivos `.js`, `.jsx`, `.ts`, `.tsx` modificados serão verificados pelo ESLint e formatados pelo Prettier
- Apenas arquivos no stage são processados, não o projeto inteiro
- Correções são aplicadas automaticamente antes do commit

## 🛠️ Configuração do Editor

### VS Code

Recomenda-se instalar as extensões:

- **ESLint**: `dbaeumer.vscode-eslint`
- **Prettier**: `esbenp.prettier-vscode`

**Configuração (`.vscode/settings.json`):**

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[javascriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

## 🔍 Solução de Problemas

### Husky não executa

Se o Husky não estiver executando, reinstale os hooks:

```bash
yarn prepare
```

### Conflitos entre ESLint e Prettier

O `eslint-config-prettier` está configurado para desabilitar regras do ESLint que conflitam com o Prettier.

### Erros de tipo TypeScript

O ESLint não substitui o TypeScript. Erros de tipo devem ser corrigidos no código:

```bash
# Verificar erros de tipo
npx tsc --noEmit
```

### Pular hooks (não recomendado)

Se necessário, você pode pular os hooks:

```bash
# Pular pre-commit
git commit --no-verify -m "mensagem"

# Pular commit-msg
HUSKY=0 git commit -m "mensagem"
```

⚠️ **Atenção:** Não é recomendado pular os hooks regularmente, pois eles garantem a qualidade do código.

## 📦 Scripts disponíveis

```json
{
  "lint": "next lint",
  "lint:fix": "next lint --fix",
  "format": "prettier --write .",
  "format:check": "prettier --check .",
  "test": "jest",
  "test:watch": "jest --watch",
  "test:coverage": "jest --coverage",
  "prepare": "husky"
}
```

## 🎯 Boas Práticas

1. **Sempre execute os testes antes de commitar**

   ```bash
   yarn test
   ```

2. **Use mensagens de commit descritivas e seguindo o padrão**

   ```bash
   git commit -m "feat: adiciona validação de email no formulário"
   ```

3. **Corrija warnings do ESLint**

   Warnings não bloqueiam o commit, mas devem ser corrigidos.

4. **Não ignore regras sem justificativa**

   Se precisar ignorar uma regra do ESLint, adicione um comentário explicando:

   ```typescript
   // eslint-disable-next-line @typescript-eslint/no-explicit-any
   const data: any = unknownData // Temporário: tipo será definido na próxima sprint
   ```

5. **Mantenha o código formatado**

   Execute `yarn format` periodicamente ou configure o editor para formatar ao salvar.

## 📚 Referências

- [ESLint Documentation](https://eslint.org/docs/latest/)
- [Prettier Documentation](https://prettier.io/docs/en/)
- [Husky Documentation](https://typicode.github.io/husky/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [lint-staged Documentation](https://github.com/lint-staged/lint-staged)
