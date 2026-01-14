# Testing Guide

## 🧪 Test Setup

Este projeto usa **Jest** e **React Testing Library** para testes unitários.

### Ferramentas de Teste

- **Jest**: Framework de testes JavaScript
- **React Testing Library**: Testes focados no comportamento do usuário
- **@testing-library/jest-dom**: Matchers customizados para assertions
- **@testing-library/user-event**: Simulação de interações do usuário

## 📦 Dependências Instaladas

```json
{
  "devDependencies": {
    "jest": "^30.2.0",
    "@testing-library/react": "^16.3.1",
    "@testing-library/jest-dom": "^6.9.1",
    "@testing-library/user-event": "^14.6.1",
    "@testing-library/dom": "^10.4.1",
    "jest-environment-jsdom": "^30.2.0",
    "@types/jest": "^30.0.0"
  }
}
```

## 🔧 Configuração

### jest.config.js

- Configuração personalizada para Next.js
- Mapeamento de módulos (`@/*`)
- Cobertura de código
- Test environment: jsdom

### jest.setup.ts

- Importa matchers do `@testing-library/jest-dom`
- Configurações globais antes dos testes

## 🏃 Como Rodar os Testes

### Comandos disponíveis:

```bash
# Rodar todos os testes
yarn test

# Rodar testes em modo watch (re-executa ao salvar)
yarn test:watch

# Rodar testes com relatório de cobertura
yarn test:coverage
```

### Rodar testes específicos:

```bash
# Testar apenas um arquivo
yarn test Hero.test.tsx

# Testar com padrão
yarn test --testPathPattern=components

# Rodar apenas testes que falharam
yarn test --onlyFailures
```

## 📁 Estrutura de Testes

```
components/
├── Hero.tsx
├── Header.tsx
├── Footer.tsx
├── Contact.tsx
├── BlogPreview.tsx
└── __tests__/
    ├── Hero.test.tsx
    ├── Header.test.tsx
    ├── Footer.test.tsx
    ├── Contact.test.tsx
    └── BlogPreview.test.tsx
```

## ✅ Componentes Testados

### 1. Hero Component

- ✅ Renderiza nome e título
- ✅ Exibe resumo profissional
- ✅ Botões CTA com links corretos
- ✅ Indicador de scroll
- ✅ Classes de estilo corretas

### 2. Header Component

- ✅ Links de navegação
- ✅ Logo/título do site
- ✅ Estrutura semântica HTML
- ✅ Atributos href corretos

### 3. Footer Component

- ✅ Copyright com ano atual
- ✅ Links de redes sociais
- ✅ Estrutura semântica HTML

### 4. Contact Component

- ✅ Seção de contato
- ✅ Links para LinkedIn e GitHub
- ✅ Link de email
- ✅ Estrutura semântica HTML

### 5. BlogPreview Component

- ✅ Lista de posts
- ✅ Títulos e excerpts
- ✅ Links para posts individuais
- ✅ Tags dos posts
- ✅ Datas formatadas
- ✅ Link "Ver todos"

## 📊 Cobertura de Código

Para gerar relatório de cobertura:

```bash
yarn test:coverage
```

Relatório será gerado em:

- `coverage/lcov-report/index.html` (visualização HTML)
- `coverage/lcov.info` (formato LCOV)

### Metas de cobertura:

- **Statements**: 80%+
- **Branches**: 75%+
- **Functions**: 80%+
- **Lines**: 80%+

## 🎯 Boas Práticas

### 1. Teste o comportamento, não a implementação

```typescript
// ✅ BOM - testa o que o usuário vê
expect(screen.getByText('Gustavo Tsuji')).toBeInTheDocument()

// ❌ RUIM - testa detalhes de implementação
expect(component.state.name).toBe('Gustavo')
```

### 2. Use queries semânticas

```typescript
// ✅ BOM - acessibilidade first
screen.getByRole('button', { name: /submit/i })
screen.getByLabelText(/email/i)

// ❌ RUIM - queries frágeis
screen.getByTestId('submit-btn')
screen.getByClassName('email-input')
```

### 3. Teste casos de uso reais

```typescript
it('allows user to submit contact form', async () => {
  const user = userEvent.setup()
  render(<ContactForm />)

  await user.type(screen.getByLabelText(/name/i), 'John Doe')
  await user.type(screen.getByLabelText(/email/i), 'john@example.com')
  await user.click(screen.getByRole('button', { name: /submit/i }))

  expect(screen.getByText(/message sent/i)).toBeInTheDocument()
})
```

### 4. Use mocks quando necessário

```typescript
// Mock de função
jest.mock('@/lib/posts', () => ({
  getAllPosts: jest.fn(() => [...])
}))

// Mock de API
jest.mock('node-fetch')
```

### 5. Organize com describe e it

```typescript
describe('Hero Component', () => {
  describe('when user is logged in', () => {
    it('shows personalized greeting', () => {
      // test
    })
  })

  describe('when user is logged out', () => {
    it('shows generic greeting', () => {
      // test
    })
  })
})
```

## 🐛 Debugging Testes

### Ver output do componente:

```typescript
const { debug } = render(<Hero />)
debug() // Imprime HTML no console
```

### Ver queries disponíveis:

```typescript
screen.debug()
screen.logTestingPlaygroundURL() // Link para Testing Playground
```

### Rodar em modo debug:

```bash
node --inspect-brk node_modules/.bin/jest --runInBand
```

## 📚 Matchers Úteis

### jest-dom matchers:

```typescript
expect(element).toBeInTheDocument()
expect(element).toBeVisible()
expect(element).toHaveClass('className')
expect(element).toHaveAttribute('href', '/page')
expect(element).toHaveTextContent('text')
expect(element).toBeDisabled()
expect(element).toBeChecked()
```

### Jest matchers:

```typescript
expect(value).toBe(expected)
expect(value).toEqual(expected)
expect(array).toContain(item)
expect(string).toMatch(/pattern/)
expect(fn).toHaveBeenCalled()
expect(fn).toHaveBeenCalledWith(args)
```

## 🔗 Recursos

- [Jest Documentation](https://jestjs.io/)
- [React Testing Library](https://testing-library.com/react)
- [Testing Library Queries](https://testing-library.com/docs/queries/about)
- [Common Mistakes](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
- [Testing Playground](https://testing-playground.com/)

## 🚀 CI/CD Integration

Para integrar com GitHub Actions:

```yaml
- name: Run tests
  run: yarn test --coverage

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage/lcov.info
```

## 📝 Próximos Passos

- [ ] Adicionar testes para páginas (app/page.tsx, app/about/page.tsx)
- [ ] Adicionar testes de integração
- [ ] Configurar coverage badges
- [ ] Adicionar testes E2E com Playwright/Cypress
- [ ] Configurar visual regression testing

---

**Configurado em**: Janeiro 2026  
**Cobertura atual**: ~80% dos componentes
