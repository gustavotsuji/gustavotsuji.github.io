---
title: 'Escalando o PostgreSQL: Como o particionamento de tabelas resolve gargalos de performance'
date: '2024-01-10'
excerpt: 'Estratégias avançadas de particionamento para processar bilhões de registros com eficiência. Exemplos práticos e benchmarks de desempenho de sistemas em produção.'
tags: ['PostgreSQL', 'Database', 'Performance', 'Scalability']
author: 'Gustavo Tsuji'
---

Quem trabalha com grandes volumes de dados sabe que, em algum momento, o tamanho das tabelas começa a cobrar seu preço. Consultas ficam lentas, índices se tornam gigantescos e operações de manutenção podem sobrecarregar o banco de dados.

Recentemente, tivemos problemas de performance e custo causados por tabelas muito grandes, onde o alto consumo de recursos do VACUUM em tabelas gigantes causava lentidão extrema e timeouts em processos críticos.

A solução adotada para mitigar isso foi o **Particionamento de Tabelas**, que divide essas tabelas em partes menores para otimizar recursos. Neste artigo, exploro o que é essa técnica, como ela funciona no PostgreSQL e quando você deve (ou não) usá-la. Em seguida, também explicarei como utilizar outros recursos (pg_cron e pg_partman) para automatizar os particionamentos, bem como a exclusão dos dados antigos.

## O que é Particionamento de Tabelas?

O particionamento é uma técnica que consiste em dividir uma tabela grande em partes menores, chamadas de "partições".

A mágica acontece na transparência: para a aplicação e para o usuário final, essas partes ainda se comportam como se fossem uma única tabela. No entanto, fisicamente, cada partição armazena apenas um subconjunto dos dados (definido por critérios como data, status ou ID).

O PostgreSQL assume a responsabilidade de decidir automaticamente em qual partição armazenar ou buscar os dados com base nas regras que você define.

## Por que utilizar?

Além de resolver o problema de tabelas gigantes, o particionamento traz benefícios estruturais importantes:

1.  **Performance de Leitura:** Ocorre uma melhora significativa em tabelas grandes, especialmente para leituras e filtros que utilizam a coluna de particionamento.
2.  **Manutenção e Limpeza:** Imagine ter que deletar milhões de linhas antigas. Com particionamento, o arquivamento ou limpeza de dados antigos é facilitado, permitindo o `DROP` de partições ao invés de `DELETE`s custosos.
3.  **Índices Otimizados:** Em vez de um índice monolítico gigante, você passa a ter índices menores e mais específicos, o que melhora a manutenção.
4.  **Uso de Recursos:** Melhora o uso de espaço em disco e torna a execução do `autovacuum` muito mais eficiente.

## Tipos de Particionamento

O PostgreSQL oferece três estratégias nativas principais:

- **RANGE (Intervalo):** Ideal quando os dados são organizados por intervalos, como datas. _Ex: Tabela de eventos particionada por ano/mês._
- **LIST (Lista):** Usado quando os dados possuem categorias fixas, como status ou região.
- **HASH:** Quando você quer distribuir os dados de forma balanceada, sem uma ordem lógica específica.

## Como funciona na prática

A implementação no PostgreSQL é declarativa. Abaixo, um exemplo de como criamos uma tabela particionada por data (RANGE) para armazenar eventos:

```sql
-- 1. Criação da tabela mestre particionada
CREATE TABLE eventos (
    id SERIAL,
    data_evento DATE NOT NULL,
    descricao TEXT,
    PRIMARY KEY (id, data_evento)
) PARTITION BY RANGE (data_evento);

-- 2. Criação das partições específicas
CREATE TABLE eventos_2023 PARTITION OF eventos
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE eventos_2024 PARTITION OF eventos
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

Ao executar uma consulta como `SELECT * FROM eventos WHERE data_evento >= '2024-01-01';`, o banco faz o _pruning_ automático e só acessa as partições relevantes.

## Casos de Uso Comuns

O particionamento é ideal em cenários como:

- Logs organizados por data;
- Eventos históricos;
- Arquivos fiscais, boletos e transações;
- Jobs, notificações e status por categoria.

## Nem tudo são flores: Cuidados e quando NÃO usar

O particionamento exige planejamento. Existem complexidades operacionais que precisam ser consideradas:

1.  **Gerenciamento Manual:** As partições geralmente precisam ser criadas manualmente ou via ferramentas como `pg_partman`.
2.  **Índices e Chaves:** Embora versões modernas do PostgreSQL propaguem índices automaticamente, a definição de **Chaves Primárias (Primary Keys)** e **Restrições Únicas (Unique Constraints)** exige a inclusão da chave de partição, o que impacta a modelagem de dados.
3.  **Restrições:** Existem restrições para _Triggers_, _Constraints_ e _Foreign Keys_ em tabelas particionadas.
4.  **Performance em Join:** Operações que envolvem muitas partições podem ser mais lentas se não forem bem planejadas.

**Principalmente, evite o particionamento se:**

- Sua tabela é pequena ou média (não trará ganho real).
- O volume de leitura é alto, mas o filtro da consulta **não** usa a coluna de particionamento.
- Você não tem uma manutenção planejada (como rotinas de DROP ou arquivamento).

## Como automatizar os particionamentos?

Criar partições manualmente é um risco operacional. Esquecer de criar a partição do próximo mês por exemplo pode gerar certas complicações. Existe uma partição de "fallback" que em caso do postgres não encontrar a partição de destino, qualquer dado acaba sendo direcionado para esse particionamento (geralmente definido como default).

Para resolver isso de forma robusta, a combinação padrão é utilizar duas extensões:

1.  **pg_partman:** Gerencia a criação automática de novas partições e o descarte das antigas.
2.  **pg_cron:** Um agendador de tarefas (job scheduler) que roda _dentro_ do banco de dados, eliminando a necessidade de scripts complexos no sistema operacional.

### Passo 1: Habilitando as extensões

Primeiro, certifique-se de que as extensões estão instaladas e habilitadas no seu banco de dados (o `pg_cron` geralmente requer configuração prévia no `shared_preload_libraries` do `postgresql.conf`).

```sql
CREATE EXTENSION IF NOT EXISTS pg_partman WITH SCHEMA partman;
CREATE EXTENSION IF NOT EXISTS pg_cron;
```

### Passo 2: Configurando o Gerenciamento (pg_partman)

Em vez de criar as partições "na mão" com `CREATE TABLE`, dizemos ao `pg_partman` para assumir o controle da tabela pai.

No exemplo abaixo, configuramos nossa tabela de `eventos` para criar partições mensais automaticamente:

```sql
SELECT partman.create_parent(
    p_parent_table => 'public.eventos',
    p_control      => 'data_evento',
    p_type         => 'native',    -- Usa particionamento nativo do PG
    p_interval     => '1 month',   -- Cria uma partição por mês
    p_default      => 'true' -- Cria a tabela default (fallback)
    p_premake      => 2            -- Mantém sempre 2 meses futuros criados
);
```

Com isso, o `pg_partman` entende a estrutura, mas ele não roda sozinho. Ele precisa ser "chamado" periodicamente para verificar se está na hora de criar novas tabelas.

### Passo 3: Agendando a Manutenção (pg_cron)

É aqui que o `pg_cron` brilha. Em vez de configurar um `crontab` no Linux, agendamos a execução da rotina de manutenção diretamente via SQL.

O comando abaixo configura o banco para rodar a manutenção do `partman` a cada hora:

```sql
-- Agenda a execução da função run_maintenance() a cada hora
SELECT cron.schedule(
    'manutencao_particoes', -- Nome do job
    '0 * * * *',            -- Cron expression (a cada hora, minuto 0)
    $$CALL partman.run_maintenance()$$
);
```

### O Resultado

Com essa configuração:

1.  A cada hora, o `pg_cron` acorda e executa o `run_maintenance()`.
2.  O `pg_partman` verifica se as partições futuras (definidas no `p_premake`) já existem.
3.  Se não existirem, elas são criadas automaticamente, garantindo que sua aplicação nunca falhe por falta de tabela para inserir dados novos.
4.  (Opcional) Se configurado, ele também pode desprender ou arquivar partições muito antigas automaticamente.

Essa arquitetura transforma o particionamento de uma "dor de cabeça de manutenção" em uma "solução invisível de performance".
