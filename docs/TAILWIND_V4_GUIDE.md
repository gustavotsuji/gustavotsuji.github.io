# Tailwind CSS 4 - Guia de Configuração

## 🎨 Arquitetura do Tailwind 4

O Tailwind CSS 4 mudou radicalmente sua arquitetura, adotando uma abordagem **CSS-first** em vez da configuração JavaScript tradicional.

### Principais Mudanças

1. **CSS-first Configuration**: Toda configuração agora é feita via CSS usando `@theme`
2. **@tailwindcss/postcss**: Plugin PostCSS separado
3. **Sem tailwind.config.js**: Não é mais necessário (mas ainda suportado para compatibilidade)
4. **@import "tailwindcss"**: Nova sintaxe de import

## 📁 Estrutura de Arquivos

```
projeto/
├── app/
│   └── globals.css          # Configuração CSS-first
├── postcss.config.js        # Plugin @tailwindcss/postcss
└── package.json
```

## 🔧 Configuração

### 1. postcss.config.js

```javascript
module.exports = {
  plugins: {
    '@tailwindcss/postcss': {}, // Plugin do Tailwind 4
    autoprefixer: {},
  },
}
```

### 2. app/globals.css

```css
@import 'tailwindcss';

@theme {
  /* Custom colors */
  --color-primary-50: #f0f9ff;
  --color-primary-100: #e0f2fe;
  /* ... outras cores */
}

/* Custom styles */
body {
  font-family: var(--font-geist-sans);
}
```

## 🎨 Definindo Cores Customizadas

No Tailwind 4, cores são definidas como CSS custom properties dentro do bloco `@theme`:

```css
@theme {
  /* Primary color palette */
  --color-primary-50: #f0f9ff;
  --color-primary-100: #e0f2fe;
  --color-primary-200: #bae6fd;
  --color-primary-300: #7dd3fc;
  --color-primary-400: #38bdf8;
  --color-primary-500: #0ea5e9;
  --color-primary-600: #0284c7;
  --color-primary-700: #0369a1;
  --color-primary-800: #075985;
  --color-primary-900: #0c4a6e;
  --color-primary-950: #082f49;
}
```

**Uso nos componentes:**

```tsx
<button className="bg-primary-600 hover:bg-primary-700 text-white">Click me</button>
```

## 📦 Dependências

```json
{
  "devDependencies": {
    "@tailwindcss/postcss": "^4.1.18",
    "tailwindcss": "^4.1.18",
    "autoprefixer": "^10.4.23"
  },
  "dependencies": {
    "@tailwindcss/typography": "^0.5.19"
  }
}
```

## ✅ Vantagens do Tailwind 4

1. **Performance**: Engine Oxide (Rust) - 10x mais rápido
2. **Simplicidade**: Menos arquivos de configuração
3. **CSS-first**: Mais próximo dos padrões web
4. **Hot Module Replacement**: Recarregamento instantâneo
5. **Menor bundle**: Otimização automática

## 🚀 Build e Deploy

```bash
# Desenvolvimento
yarn dev

# Build de produção
yarn build

# Limpar cache
rm -rf .next && yarn dev
```

## 📚 Documentação Oficial

- [Tailwind CSS 4 Beta](https://tailwindcss.com/docs/v4-beta)
- [@tailwindcss/postcss](https://github.com/tailwindlabs/tailwindcss/tree/next/packages/%40tailwindcss-postcss)
- [Migration Guide](https://tailwindcss.com/docs/upgrade-guide)

## 🎯 Dicas Importantes

1. **Não use `@tailwind` directives**: Use `@import "tailwindcss"` instead
2. **Não use `@apply` em produção**: Prefira classes utilitárias
3. **CSS custom properties**: Use `--color-*` para cores customizadas
4. **Limpe o cache**: Sempre delete `.next` após mudanças de configuração
5. **PostCSS é obrigatório**: O plugin `@tailwindcss/postcss` é essencial

## 🐛 Troubleshooting

### Cores não aparecem

- Verifique se o `@theme` está correto no `globals.css`
- Limpe o cache: `rm -rf .next`
- Reinicie o servidor

### Build falha

- Verifique se `postcss.config.js` usa CommonJS (module.exports)
- Confirme que `@tailwindcss/postcss` está instalado
- Verifique erros no console

### Estilos não aplicam

- Hard refresh: Cmd+Shift+R (Mac) ou Ctrl+Shift+R (Windows)
- Limpe `.next` e `node_modules/.cache`
- Verifique se `@import "tailwindcss"` está no topo do CSS

---

**Data da configuração**: Janeiro 2026  
**Versão**: Tailwind CSS 4.1.18
