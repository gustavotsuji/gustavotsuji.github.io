---
title: 'Por que desistimos de migrar para SQS Standard'
date: '2026-02-06'
excerpt: 'Como uma análise de dados nos salvou de um over-engineering clássico: a história de quando decidimos não migrar do SQS FIFO para Standard.'
tags: ['AWS', 'SQS', 'Architecture', 'Over-engineering', 'Trade-offs']
author: 'Gustavo Tsuji'
---

Recentemente, nosso time de engenharia se deparou com um desafio clássico: modernização de arquitetura. Nossas filas SQS FIFO, essenciais para a integridade dos dados entre microsserviços, pareciam estar se tornando um gargalo teórico. O plano era migrar para SQS Standard, ganhando throughput ilimitado.

Fizemos o _Technical Design Document_, desenhamos a solução de idempotência e estratégias de _locking_. Mas, na etapa final de validação, decidimos **abortar a missão**. Este artigo explica por que a melhor decisão foi ficar onde estávamos.

**A "Escala Infinita"**
O SQS Standard é poderoso. Ele remove o teto de 300 TPS do FIFO e elimina o _Head-of-Line Blocking_ (onde uma mensagem com erro trava o grupo). No papel, migrar parecia ser o caminho certo para a redução do processamento de uma etapa em um dos fluxos do time.

Mas o novo desenho exigia que nossa aplicação assumisse responsabilidades pesadas:

1. **Ordem:** O banco de dados teria que rejeitar versões antigas de mensagens (_Optimistic Locking_).
2. **Unicidade:** Precisaríamos de um cache distribuído (Redis) para evitar processamento duplicado.

**A Solução: SQS Standard + Inteligência na Aplicação**
Migrar para Standard desbloqueia escalabilidade elástica e concorrência massiva, mas muda a semântica para "At-Least-Once" e "Best-Effort Ordering". Teríamos que implementar na aplicação o que a AWS fazia por nós.

**1. Tratamento de Ordem (Optimistic Locking)**
Como as mensagens podem chegar fora de ordem (v2 antes de v1), é necessário que haja algum mecanismo de controle das versões conforme as mensagens são processadas, como um banco de dados. Poderíamos adotar _Guard Clauses_ nos updates SQL:

```sql
UPDATE orders
SET status = :new_status, version = :new_version
WHERE id = :order_id AND version < :new_version;

```

Se `Rows affected = 0`, sabemos que a mensagem é obsoleta (stale) e fazemos o _ack_ silenciosamente. Isso também protege contra condições de corrida onde um evento de "Criação" atrasado tenta sobrescrever uma "Deleção" (Soft Delete).

**2. Idempotência e Deduplicação**
O SQS Standard pode entregar a mesma mensagem mais de uma vez. Para evitar duplicidade, adotamos uma estratégia em duas camadas:

1. **Cache Distribuído (Redis):** Um bloqueio curto (`SETNX` com TTL de 30s) no ID da mensagem para evitar que dois _workers_ processem o mesmo evento simultaneamente.
2. **Idempotência de Negócio:** Operações desenhadas para serem idempotentes (ex: `SET x = 10` em vez de `x = x + 10`).
   No caso do SQS FIFO, existe o `MessageDeduplication` que já permite que sejam feitas checagens de deduplicação.

**O Choque de Realidade: Dados vs. Suposições**
Antes de iniciar a refatoração, fizemos uma análise das métricas do CloudWatch. A pergunta era: _"Quantas vezes, nos últimos 6 meses, fomos limitados pelo teto do FIFO?"_

A resposta: **Praticamente nunca.**

Nossos picos de tráfego, mesmo em momentos sazonais, raramente extrapolam o limite de 3.000 TPS (com batching). O "gargalo" que estávamos tentando resolver era hipotético.

**A Equação do Custo de Complexidade**
Engenharia de software é sobre _trade-offs_.

- **Ficar no FIFO:** Custo de desenvolvimento zero. Complexidade cognitiva baixa. Risco de _throttling_ em picos extremos (muito raros).
- **Migrar para Standard:** Custo de desenvolvimento alto. Complexidade cognitiva alta (lidar com _race conditions_, _duplicates_, _out-of-order_). Risco de bugs de concorrência.

Percebemos que estávamos prestes a cometer **Over-engineering**. Estávamos resolvendo um problema de escala do Google ou da Netflix, quando nosso cenário atual e de médio prazo é perfeitamente atendido pelas garantias "out-of-the-box" da AWS.

**Conclusão**
Manter o FIFO não foi um ato de preguiça, mas de estratégia. Escolhemos gastar nossos "tokens de inovação" em funcionalidades que trazem valor direto ao cliente, e não em reescrever uma infraestrutura que funciona.

Se um dia o tráfego aumentar 10x, o design técnico de migração já está documentado e pronto na gaveta. Até lá, seguimos com a simplicidade.
