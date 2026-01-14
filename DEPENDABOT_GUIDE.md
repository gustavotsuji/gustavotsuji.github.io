# Dependabot Configuration Guide

## 📦 O que é o Dependabot?

O Dependabot é uma ferramenta do GitHub que automaticamente:

- Verifica atualizações de dependências
- Cria Pull Requests com as atualizações
- Mantém o projeto seguro e atualizado
- Detecta vulnerabilidades de segurança

## 🔧 Configuração Atual

### NPM/Yarn Dependencies

- **Frequência**: Toda segunda-feira às 09:00 (horário de São Paulo)
- **Limite de PRs**: 10 simultâneos
- **Agrupamentos**:
  - **Next.js**: `next`, `react`, `react-dom`
  - **Tailwind CSS**: `tailwindcss`, `@tailwindcss/*`, `autoprefixer`, `postcss`
  - **Types**: Todos `@types/*`
  - **Dev Dependencies**: Dependências de desenvolvimento

### GitHub Actions

- **Frequência**: Toda segunda-feira às 09:00 (horário de São Paulo)
- **Limite de PRs**: 5 simultâneos
- **Agrupamento**: Todas as actions juntas

## 📋 Formato dos Commits

### Dependencies (npm/yarn)

```
chore(deps): update next to v16.1.2
chore(deps-dev): update @types/node to v25.1.0
```

### GitHub Actions

```
chore(ci): update actions/checkout to v4.2.0
```

## 🏷️ Labels Aplicadas

Todos os PRs do Dependabot terão as seguintes labels:

- `dependencies` - Indica atualização de dependência
- `automated` - PR criado automaticamente
- `github-actions` - Específico para actions (apenas actions)

## 📊 Grupos de Atualização

### Por que agrupar?

Agrupar dependências relacionadas evita múltiplos PRs pequenos e facilita a revisão.

**Exemplo**: Em vez de 3 PRs separados para `next`, `react` e `react-dom`, você receberá apenas 1 PR com todas as atualizações do Next.js juntas.

## 🚀 Como Funciona

1. **Segunda-feira 09:00**: Dependabot verifica atualizações
2. **Cria PRs**: Um PR para cada grupo com atualizações disponíveis
3. **Testes automáticos**: GitHub Actions roda os testes
4. **Revisão**: Você revisa e aprova (ou o PR é auto-merged se configurado)
5. **Merge**: Dependências atualizadas! 🎉

## 📝 Personalizações Opcionais

### Adicionar Reviewers

Descomente e adicione seu username:

```yaml
reviewers:
  - 'gustavotsuji'
```

### Adicionar Assignees

```yaml
assignees:
  - 'gustavotsuji'
```

### Ignorar Dependências Específicas

```yaml
ignore:
  - dependency-name: 'next'
    versions: ['15.x'] # Ignora versão 15.x
```

### Auto-merge (Requer configuração adicional)

Para auto-merge de atualizações seguras (patch/minor), você pode:

1. Habilitar auto-merge no GitHub
2. Adicionar workflow para aprovar PRs do Dependabot automaticamente

## 🔒 Segurança

O Dependabot também:

- ✅ Detecta vulnerabilidades de segurança
- ✅ Cria PRs prioritários para correções de segurança
- ✅ Mantém o projeto atualizado com patches de segurança

## 📚 Recursos Úteis

- [Documentação oficial do Dependabot](https://docs.github.com/en/code-security/dependabot)
- [Opções de configuração](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file)
- [Grupos de atualização](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file#groups)

## 🎯 Próximos Passos

1. ✅ Commit e push do `.github/dependabot.yml`
2. ⏳ Aguardar primeira execução (próxima segunda-feira)
3. 📝 Revisar e mergear os PRs criados
4. 🔄 Repetir semanalmente de forma automática

---

**Configurado em**: Janeiro 2026  
**Timezone**: America/Sao_Paulo  
**Frequência**: Semanal (Segunda-feira 09:00)
