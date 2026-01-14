# Como Criar Posts no Blog

Este guia explica como adicionar novos artigos ao seu blog.

## 📝 Estrutura de um Post

Os posts são arquivos Markdown (`.md`) localizados na pasta `content/posts/`.

### Template Básico

Crie um novo arquivo em `content/posts/` com o seguinte formato:

```markdown
---
title: 'Título do Seu Post'
date: '2024-01-15'
excerpt: 'Uma breve descrição do post que aparecerá na listagem. Deve ter 1-2 linhas.'
tags: ['Tag1', 'Tag2', 'Tag3']
author: 'Gustavo Tsuji'
---

# Título do Seu Post

Seu conteúdo começa aqui...

## Seção 1

Texto da seção...

### Subseção

Mais conteúdo...

## Código

\`\`\`javascript
const exemplo = 'código aqui';
console.log(exemplo);
\`\`\`

## Conclusão

Finalize seu artigo...

---

_Comentários finais ou call-to-action_
```

## 🎯 Frontmatter Explicado

O frontmatter (entre `---`) contém metadados do post:

| Campo     | Tipo   | Obrigatório | Descrição                             |
| --------- | ------ | ----------- | ------------------------------------- |
| `title`   | string | ✅          | Título do post                        |
| `date`    | string | ✅          | Data no formato YYYY-MM-DD            |
| `excerpt` | string | ✅          | Resumo curto (1-2 linhas)             |
| `tags`    | array  | ✅          | Lista de tags/categorias              |
| `author`  | string | ❌          | Nome do autor (padrão: Gustavo Tsuji) |

## 📂 Nomeando Arquivos

Use kebab-case para nomear os arquivos:

✅ **Bom:**

- `optimizing-aws-costs.md`
- `circuit-breakers-nodejs.md`
- `postgresql-partitioning.md`

❌ **Evite:**

- `Optimizing AWS Costs.md` (espaços)
- `OptimizingAWSCosts.md` (camelCase)
- `optimizing_aws_costs.md` (underscore)

O nome do arquivo será usado na URL: `/blog/optimizing-aws-costs`

## ✍️ Formatação Markdown

### Títulos

```markdown
# H1 - Título Principal (use apenas uma vez)

## H2 - Seção

### H3 - Subseção

#### H4 - Sub-subseção
```

### Texto

```markdown
**negrito**
_itálico_
`código inline`
[link](https://exemplo.com)
```

### Listas

```markdown
- Item 1
- Item 2
  - Sub-item 2.1
  - Sub-item 2.2

1. Primeiro
2. Segundo
3. Terceiro
```

### Blocos de Código

````markdown
```javascript
const hello = 'world'
console.log(hello)
```

```typescript
interface User {
  name: string
  age: number
}
```

```bash
npm install
npm run dev
```
````

### Citações

```markdown
> Esta é uma citação.
> Pode ter múltiplas linhas.
```

### Imagens

```markdown
![Texto alternativo](/images/exemplo.png)
```

### Tabelas

```markdown
| Coluna 1 | Coluna 2 | Coluna 3 |
| -------- | -------- | -------- |
| Dado 1   | Dado 2   | Dado 3   |
| Dado 4   | Dado 5   | Dado 6   |
```

## 📋 Exemplo Completo

Veja os exemplos em `content/posts/`:

- `optimizing-aws-costs.md` - Post técnico sobre AWS
- `circuit-breakers-nodejs.md` - Tutorial com código
- `postgresql-partitioning.md` - Artigo sobre banco de dados

## 🚀 Workflow para Criar um Post

1. **Crie o arquivo:**

   ```bash
   touch content/posts/meu-novo-post.md
   ```

2. **Adicione o frontmatter e conteúdo:**

   ```markdown
   ---
   title: 'Meu Novo Post'
   date: '2024-01-15'
   excerpt: 'Descrição breve do post'
   tags: ['Node.js', 'Tutorial']
   author: 'Gustavo Tsuji'
   ---

   # Meu Novo Post

   Conteúdo aqui...
   ```

3. **Visualize localmente:**

   ```bash
   yarn dev
   ```

   Acesse `http://localhost:3000/blog`

4. **O post aparecerá automaticamente** na listagem do blog! 🎉

## 🏷️ Tags Recomendadas

Use tags consistentes para melhor organização:

**Linguagens:**

- JavaScript, TypeScript, Python, Java, Go

**Tecnologias:**

- Node.js, React, Next.js, PostgreSQL, MongoDB

**Conceitos:**

- Architecture, Performance, Security, DevOps, Testing

**Cloud:**

- AWS, Azure, GCP, Docker, Kubernetes

**Práticas:**

- Best Practices, Tutorial, Case Study, Guide

## 💡 Dicas de Escrita

1. **Título claro** - Descreva o que o leitor aprenderá
2. **Introdução forte** - Explique o problema ou contexto
3. **Código comentado** - Adicione comentários explicativos
4. **Exemplos práticos** - Use casos reais quando possível
5. **Conclusão** - Resuma os pontos principais
6. **Call-to-action** - Incentive discussão ou feedback

## 🔄 Atualizando Posts

Para atualizar um post existente:

1. Edite o arquivo `.md` correspondente
2. O site será atualizado automaticamente (em dev mode)
3. Commit as mudanças no Git

## ❓ Troubleshooting

**Post não aparece na listagem?**

- Verifique se o frontmatter está correto
- Confirme que o arquivo está em `content/posts/`
- Verifique se a extensão é `.md`

**Código não está formatado?**

- Use as crases triplas: ` ``` `
- Especifique a linguagem: ` ```javascript `

**Erro de build?**

- Valide o YAML do frontmatter
- Verifique aspas e caracteres especiais

---

## 📚 Recursos Adicionais

- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Flavored Markdown](https://github.github.com/gfm/)
- [Frontmatter Spec](https://jekyllrb.com/docs/front-matter/)

Bom blogging! ✍️
