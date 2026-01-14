# Guia de Atualização para Node.js 24

## 📋 Status Atual

- **Versão atual do Node.js**: 20.18.2
- **Versão alvo**: 24.x (LTS)

## ✅ Arquivos Atualizados

Os seguintes arquivos foram configurados para Node.js 24:

1. **.nvmrc** - Define Node.js 24 para o NVM
2. **.node-version** - Define Node.js 24 para outros gerenciadores
3. **package.json** - Adicionado `engines` especificando Node >= 24.0.0
4. **.github/workflows/deploy.yml** - Atualizado para usar Node.js 24 no CI/CD

## 🚀 Como Atualizar o Node.js

### Opção 1: Usando NVM (Recomendado)

```bash
# Instalar Node.js 24
nvm install 24

# Usar Node.js 24
nvm use 24

# Definir como padrão
nvm alias default 24

# Verificar versão
node --version  # Deve mostrar v24.x.x
```

### Opção 2: Usando Homebrew (macOS)

```bash
# Atualizar Homebrew
brew update

# Instalar Node.js 24
brew install node@24

# Linkar versão
brew link node@24 --force --overwrite

# Verificar versão
node --version
```

### Opção 3: Usando n (Node Version Manager)

```bash
# Instalar n (se não tiver)
npm install -g n

# Instalar Node.js 24
sudo n 24

# Verificar versão
node --version
```

### Opção 4: Download Manual

Baixe e instale do site oficial:

- https://nodejs.org/en/download/

## 📦 Após Atualizar o Node.js

```bash
# 1. Instalar Yarn globalmente (se ainda não tiver)
npm install -g yarn

# 2. Verificar versão do Yarn
yarn --version  # Deve ser >= 1.22.0

# 3. Limpar cache e node_modules
rm -rf node_modules yarn.lock

# 4. Reinstalar dependências com Yarn
yarn install

# 5. Testar o build
yarn build

# 6. Testar servidor de desenvolvimento
yarn dev
```

## 🔍 Verificar Compatibilidade

```bash
# Verificar versão do Node.js
node --version

# Verificar versão do Yarn
yarn --version

# Verificar dependências desatualizadas
yarn outdated

# Auditar segurança
yarn audit
```

## ✨ Benefícios do Node.js 24

- **Performance**: Melhorias de desempenho no V8 engine
- **Segurança**: Correções de segurança mais recentes
- **Features**: Novas APIs e funcionalidades
- **LTS**: Long Term Support até 2027
- **ESM**: Melhor suporte para módulos ES
- **Test Runner**: Test runner nativo melhorado

## 📝 Compatibilidade

### Pacotes Compatíveis com Node.js 24:

- ✅ Next.js 16.1.1 - Totalmente compatível
- ✅ React 19.2.3 - Totalmente compatível
- ✅ TypeScript 5.3.3 - Totalmente compatível
- ✅ Tailwind CSS 3.4.1 - Totalmente compatível
- ✅ ESLint 9.39.2 - Totalmente compatível

### Verificar Compatibilidade:

```bash
# Executar testes
yarn test

# Verificar build
yarn build

# Verificar lint
yarn lint
```

## 🐛 Troubleshooting

### Problema: "node: command not found"

```bash
# Recarregar shell
source ~/.zshrc  # ou source ~/.bashrc

# Ou reiniciar terminal
```

### Problema: Conflitos de dependências

```bash
# Limpar tudo e reinstalar
rm -rf node_modules yarn.lock
yarn cache clean
yarn install
```

### Problema: Build falha

```bash
# Verificar logs completos
yarn build --verbose

# Verificar versão do Node.js
node --version

# Deve mostrar v24.x.x
```

## 📊 Checklist de Atualização

- [x] Criado arquivo `.nvmrc` com versão 24
- [x] Criado arquivo `.node-version` com versão 24
- [x] Atualizado `package.json` com engines
- [x] Atualizado `.github/workflows/deploy.yml` para Node 24
- [ ] Instalado Yarn globalmente: `npm install -g yarn`
- [ ] Atualizado Node.js local para versão 24
- [ ] Reinstalado dependências com `yarn install`
- [ ] Testado build com `yarn build`
- [ ] Testado dev server com `yarn dev`
- [ ] Commitado mudanças no Git

## 🎯 Próximos Passos

1. **Instalar Yarn**: `npm install -g yarn`
2. **Atualizar Node.js local** usando uma das opções acima
3. **Reinstalar dependências**: `yarn install`
4. **Testar aplicação**: `yarn dev`
5. **Fazer build**: `yarn build`
6. **Commit**: `git add . && git commit -m "chore: update to Node.js 24 and Yarn"`
7. **Push**: `git push origin master`

## 📚 Recursos

- [Node.js 24 Release Notes](https://nodejs.org/en/blog/release/)
- [Node.js Compatibility](https://node.green/)
- [Next.js Node.js Requirements](https://nextjs.org/docs/pages/building-your-application/deploying#nodejs-version)

---

**Nota**: Após atualizar o Node.js, o GitHub Actions usará automaticamente Node.js 24 no deploy!
