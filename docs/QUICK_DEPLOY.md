# 🚀 Deploy Rápido - GitHub Pages

## Opção 1: Deploy Manual (Mais Simples)

### Passo a Passo:

```bash
# 1. Build do projeto
npm run build

# 2. (Opcional) Teste local
npx serve out

# 3. Deploy para GitHub Pages
npm run deploy
```

### Primeira vez? Configure o repositório:

1. Acesse: https://github.com/gustavotsuji/gustavotsuji.github.io/settings/pages
2. **Source**: Deploy from a branch
3. **Branch**: `gh-pages` → `/ (root)` → Save

✅ Seu site estará em: **https://gustavotsuji.github.io**

---

## Opção 2: Deploy Automático (GitHub Actions)

### Configure uma vez:

1. Acesse: https://github.com/gustavotsuji/gustavotsuji.github.io/settings/pages
2. **Source**: GitHub Actions
3. Salve

### Deploy automático a cada push:

```bash
git add .
git commit -m "feat: minha alteração"
git push origin master
```

🤖 GitHub Actions faz deploy automaticamente!

Acompanhe em: https://github.com/gustavotsuji/gustavotsuji.github.io/actions

---

## 📝 Comandos Úteis

```bash
# Desenvolvimento local
npm run dev

# Build para produção
npm run build

# Deploy manual
npm run deploy

# Testar build localmente
npx serve out
```

---

## 🎯 Primeiro Deploy

Execute agora mesmo:

```bash
# 1. Commit suas mudanças
git add .
git commit -m "feat: portfolio completo"

# 2. Faça push
git push origin master

# 3. Deploy
npm run deploy
```

Aguarde 2-5 minutos e acesse: **https://gustavotsuji.github.io** 🎉

---

## 🔄 Atualizar o Site

Sempre que fizer mudanças:

```bash
git add .
git commit -m "descrição da mudança"
git push origin master
npm run deploy  # Só se não usar GitHub Actions
```

---

## ✅ Verificação

Depois do deploy, teste:

- [ ] https://gustavotsuji.github.io
- [ ] Menu de navegação
- [ ] Posts do blog
- [ ] Links externos (LinkedIn, GitHub)
- [ ] Versão mobile

---

## ❓ Problemas?

**Site não aparece?**

- Aguarde 5 minutos
- Limpe cache (Ctrl+Shift+R)
- Verifique se branch `gh-pages` existe

**Erro 404 nos posts?**

- Confirme `output: 'export'` no `next.config.js`
- Rebuild: `npm run build && npm run deploy`

**Mais ajuda?**
Veja `DEPLOY.md` para guia completo.
