# 🚀 Deploy para GitHub Pages

Este guia explica como publicar seu portfolio Next.js no GitHub Pages.

## 📋 Pré-requisitos

- [x] Repositório: `gustavotsuji.github.io` (já criado)
- [x] Next.js configurado para export estático (já configurado)
- [ ] Git configurado localmente
- [ ] Conta GitHub com acesso ao repositório

## 🔧 Configuração Atual

O projeto já está configurado para GitHub Pages:

### ✅ `next.config.js`

```javascript
output: 'export' // Gera HTML estático
images: {
  unoptimized: true
} // Imagens otimizadas para static export
```

### ✅ `.nojekyll`

Arquivo criado para evitar processamento Jekyll

## 📦 Opção 1: Deploy Manual (Recomendado para primeira vez)

### Passo 1: Build do projeto

```bash
npm run build
```

Isso irá:

- Compilar o projeto Next.js
- Gerar arquivos estáticos na pasta `out/`
- Otimizar CSS, JS e assets

### Passo 2: Teste local do build

```bash
npx serve out
```

Acesse `http://localhost:3000` e verifique se tudo está funcionando.

### Passo 3: Commit e Push

```bash
# Adicione todos os arquivos do projeto (exceto node_modules, out, .next)
git add .
git commit -m "feat: portfolio completo com blog integrado"

# Faça push para o repositório
git push origin master
```

### Passo 4: Deploy para GitHub Pages

```bash
# Instale o gh-pages (uma vez)
npm install --save-dev gh-pages

# Deploy (publica a pasta out/ no branch gh-pages)
npm run deploy
```

### Passo 5: Configurar GitHub Pages

1. Acesse: https://github.com/gustavotsuji/gustavotsuji.github.io/settings/pages
2. Em **Source**, selecione: `Deploy from a branch`
3. Em **Branch**, selecione: `gh-pages` e pasta `/ (root)`
4. Clique em **Save**

🎉 Seu site estará disponível em: **https://gustavotsuji.github.io**

---

## 🤖 Opção 2: Deploy Automático com GitHub Actions

Crie o arquivo `.github/workflows/deploy.yml` (já criado) para deploy automático a cada push.

### Como funciona:

1. Você faz um commit e push no branch `master`
2. GitHub Actions automaticamente:
   - Instala dependências
   - Faz build do projeto
   - Publica no branch `gh-pages`
3. Site é atualizado automaticamente!

### Ativar GitHub Actions:

1. Acesse: https://github.com/gustavotsuji/gustavotsuji.github.io/settings/actions
2. Em **Actions permissions**, selecione: `Allow all actions and reusable workflows`
3. Salve as configurações

### Primeiro Deploy com Actions:

```bash
# Faça qualquer alteração e commit
git add .
git commit -m "feat: deploy automático configurado"
git push origin master

# GitHub Actions irá fazer o deploy automaticamente!
```

Acompanhe o progresso em:
https://github.com/gustavotsuji/gustavotsuji.github.io/actions

---

## 📝 Scripts do package.json

Já adicionados ao seu `package.json`:

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "export": "next build && next export",
    "deploy": "npm run build && gh-pages -d out"
  }
}
```

---

## 🔍 Verificação

Depois do deploy, verifique:

### ✅ Checklist:

- [ ] Site carrega: https://gustavotsuji.github.io
- [ ] Navegação funciona (Home, About, Blog, Contact)
- [ ] Posts do blog abrem corretamente
- [ ] Código tem syntax highlighting
- [ ] Links do LinkedIn e GitHub funcionam
- [ ] Imagens carregam (se houver)
- [ ] Responsivo funciona em mobile

### 🐛 Troubleshooting:

**Site não aparece após deploy?**

- Aguarde 2-5 minutos (GitHub Pages demora para atualizar)
- Verifique se o branch `gh-pages` foi criado
- Limpe cache do navegador (Ctrl+Shift+R)

**Erro 404 nas rotas do blog?**

- Verifique se `output: 'export'` está no `next.config.js`
- Certifique-se que não está usando funções SSR (getServerSideProps)

**CSS não carrega?**

- Limpe cache: `rm -rf .next out`
- Rebuild: `npm run build`

**GitHub Actions falhou?**

- Verifique logs em: `/actions`
- Confira permissões de escrita no repositório

---

## 🔄 Atualizações Futuras

### Para adicionar novo post ao blog:

1. Crie arquivo em `content/posts/nome-do-post.md`
2. Adicione frontmatter e conteúdo
3. Commit e push:
   ```bash
   git add content/posts/nome-do-post.md
   git commit -m "post: adiciona artigo sobre [tema]"
   git push origin master
   ```
4. Se usando GitHub Actions, deploy é automático!
5. Se usando deploy manual, rode: `npm run deploy`

### Para atualizar componentes/estilo:

```bash
# Faça suas alterações
git add .
git commit -m "feat: atualiza [componente]"
git push origin master

# Deploy manual (se não usar Actions)
npm run deploy
```

---

## 📊 Monitoramento

Depois do deploy, você pode:

- Ver estatísticas: GitHub > Insights > Traffic
- Monitorar deploy: GitHub > Actions (se usar CI/CD)
- Ver branch gh-pages: `git checkout gh-pages`

---

## 🎯 Próximos Passos

Depois do primeiro deploy:

1. **Domínio customizado** (opcional):
   - Compre um domínio (ex: gustavotsuji.com.br)
   - Configure CNAME no GitHub Pages
   - Aponte DNS para GitHub

2. **SEO**:
   - Adicione sitemap.xml
   - Configure robots.txt
   - Adicione meta tags Open Graph

3. **Analytics** (opcional):
   - Google Analytics
   - Plausible
   - Simple Analytics

4. **Performance**:
   - Teste no Lighthouse
   - Otimize imagens
   - Adicione cache headers

---

## 💡 Dicas

✅ **Faça build local antes de fazer push** para evitar erros em produção:

```bash
npm run build
npx serve out  # Teste localmente
```

✅ **Commits semânticos** para melhor histórico:

- `feat:` nova funcionalidade
- `fix:` correção de bug
- `post:` novo artigo
- `style:` mudanças de estilo
- `docs:` documentação

✅ **Branches** para grandes mudanças:

```bash
git checkout -b feature/nova-funcionalidade
# Faça alterações
git commit -m "feat: adiciona nova funcionalidade"
git push origin feature/nova-funcionalidade
# Crie Pull Request no GitHub
```

---

**Pronto para deploy? Execute:**

```bash
npm run build && npm run deploy
```

🚀 Boa sorte com seu portfolio!
