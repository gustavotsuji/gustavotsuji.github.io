---
title: 'Pare de quebrar o Pipeline: Como o "Shift-Left" e o Husky podem salvar o seu dia (e o orçamento da empresa)'
date: '2026-01-26'
excerpt: 'Traga validações do pipeline para a máquina do desenvolvedor com Husky — evite quebras no CI e economize tempo e recursos.'
tags: ['Husky', 'Git', 'CI/CD', 'Shift-Left', 'DevOps', 'Quality Assurance']
author: 'Gustavo Tsuji'
---

Quem nunca passou por isso: você finaliza uma feature, abre o Pull Request e espera ansiosamente pelo _green check_ do pipeline. Dez minutos depois, o CI falha. Motivo? Um erro simples de lint ou uma validação do Sonar que poderia ter sido resolvida em segundos na sua máquina.

Além da frustração, isso gera um alto volume de execuções no GitHub Actions (ou similar), elevando custos e desperdiçando recursos computacionais. Pior ainda: cada correção trivial exige um novo push, reiniciando a esteira e invalidando aprovações já obtidas no Code Review.

Neste artigo, vou compartilhar uma abordagem para trazer essas validações para o ambiente local (**Shift-Left**) utilizando o **Husky**, com uma estratégia de configuração personalizada que não trava a produtividade do time.

## O Conceito "Shift-Left"

A ideia central é simples: **e se pudéssemos executar todos os steps localmente para ter uma chance maior de sucesso e evitar a perda de tempo no fluxo do Git Actions?**.

O "Shift-Left" traz as verificações de qualidade (Lint, Testes, Sonar, Segurança) para a máquina do desenvolvedor, antes mesmo do código chegar ao servidor.

## O Guardião: Husky

Para orquestrar isso, utilizamos o **Husky**. Ele gerencia scripts acionados automaticamente por eventos do Git (Git Hooks). Ele atua como um _Gatekeeper_ (Guardião): se a verificação falhar, o comando do git (commit ou push) é abortado instantaneamente.

Dividimos a estratégia em dois momentos principais:

1.  **Pre-commit:** Roda formatações e linters apenas nos arquivos alterados (_staged_).

2.  **Pre-push:** Roda testes mais pesados e verificações de segurança antes de enviar ao repositório remoto.

## O Pulo do Gato: Customização por Desenvolvedor

Um dos maiores receios ao adotar hooks locais é a lentidão. "Vou ter que rodar o Sonar toda vez que der um push?". A resposta é: **depende de você**.

Para resolver isso, criamos uma estrutura de **Arquivo de Controle Local (Untracked)**.
O desenvolvedor cria um arquivo na raiz do projeto (ex: `.husky.user.config`) que é ignorado pelo `.gitignore`.

Neste arquivo, definimos variáveis booleanas que ligam ou desligam verificações específicas:

```bash
# .husky.user.config
# Por PADRÃO, steps podem vir desabilitados para performance.
# O dev ativa o que faz sentido para o momento dele.

export HUSKY_RUN_LINT=true           # ESLint + Prettier
export HUSKY_RUN_GITLEAKS=true       # Detecção de senhas/segredos
export HUSKY_RUN_UNIT_TESTS=true     # Testes unitários
export HUSKY_RUN_TRIVY_SCAN=true     # Scanner de vulnerabilidades
export HUSKY_RUN_SONAR_SCAN=false    # SonarQube (pesado, ativar quando necessário)

```

Os scripts do Husky foram adaptados com lógica condicional para ler esse arquivo antes da execução. Se a variável for `false` ou estiver ausente, o script pula aquela etapa.

## O Que Estamos Validando?

Com essa estrutura, conseguimos garantir diversas camadas de qualidade antes do código sair da máquina:

### No Pre-Commit:

- **Lint & Prettier:** Garante o estilo de código.

- **GitLeaks:** Verifica se você não está commitando chaves de API ou senhas acidentalmente. Se encontrar segredos, o commit é abortado.

- **Validação de Dockerfile:** Se o `Dockerfile` foi alterado, tenta realizar o build para garantir que a imagem não está quebrada.

### No Pre-Push:

- **Testes Unitários:** O básico. Se quebrou o teste, não sobe.

- **Trivy:** Um scanner de vulnerabilidades em dependências e containers.

- **NPM Audit:** Verifica vulnerabilidades de alta severidade nos pacotes.

### A Integração com SonarQube Local

Esta é a parte mais robusta. Criamos um script que, caso habilitado (`HUSKY_RUN_SONAR_SCAN=true`):

1. Verifica se o container do SonarQube está rodando via Docker. Se não estiver, ele sobe o container automaticamente.

2. Verifica se o projeto existe no Sonar local. Se não existir, ele cria o projeto automaticamente via API.

3. Executa o `sonar-scanner` via Docker.

4. Consulta o **Quality Gate** via API. Se o status for **ERROR**, o push é abortado e o desenvolvedor recebe o link do dashboard para corrigir os problemas.

## Conclusão

Embora seja possível pular essas verificações com `--no-verify` (mas sabemos que você não deveria fazer isso... 😅), o objetivo dessa arquitetura é empoderar o desenvolvedor.

Ao rodar validações localmente de forma seletiva, reduzimos drasticamente o tempo de espera no CI/CD, economizamos dinheiro com GitHub Actions e mantemos a sanidade do time evitando quebras bobas de pipeline.

E você, já usa alguma estratégia de **Shift-Left** no seu fluxo de desenvolvimento?

---

_Artigo baseado na apresentação técnica sobre Husky e automação de Git Hooks._
