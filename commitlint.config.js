module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat', // Nova funcionalidade
        'fix', // Correção de bug
        'docs', // Mudanças na documentação
        'style', // Mudanças de formatação (não afetam o código)
        'refactor', // Refatoração de código
        'perf', // Melhorias de performance
        'test', // Adição ou correção de testes
        'chore', // Mudanças em ferramentas, configurações, dependências
        'ci', // Mudanças em arquivos de CI
        'build', // Mudanças no sistema de build
        'revert', // Reversão de commit anterior
      ],
    ],
    'subject-case': [0], // Permite qualquer case no subject
    'body-max-line-length': [0], // Remove limite de linha no body
  },
}
