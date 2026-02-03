---
title: 'Zero Downtime e Economia na Nuvem: Como o Blue/Green Deployment no RDS salvou nosso orçamento'
date: '2026-01-26'
excerpt: 'Relato prático de como usamos Blue/Green Deployment no AWS RDS para migrar e redimensionar um banco PostgreSQL com zero downtime, reduzindo custos operacionais.'
tags: ['AWS', 'RDS', 'PostgreSQL', 'Blue-Green', 'Database', 'Cloud Cost']
author: 'Gustavo Tsuji'
---

# Zero Downtime e Economia na Nuvem: Como o Blue/Green Deployment no RDS salvou nosso orçamento

Quem trabalha com bancos de dados em produção conhece o frio na barriga de realizar manutenções críticas. Recentemente, nos deparamos com um cenário comum, mas desafiador: precisávamos atualizar e redimensionar um banco de dados PostgreSQL vital, que recebia centenas de operações por minuto, sem impactar a experiência do usuário.

O cenário era o seguinte: nossa base de dados (o _taffarel_) estava subutilizada. Análises mostraram que poderíamos reduzir a CPU pela metade, migrando de uma instância `r6g.4xlarge` para `x2g.2xlarge`, gerando uma **economia mensal de $425**.

Além disso, enfrentávamos a questão do "End of Life" das versões do PostgreSQL na AWS. Manter versões antigas exige a contratação de suporte estendido (que custa cerca de **$413/mês**), um custo que só serve como solução temporária.

A pergunta de um milhão de dólares era: **Como fazer essa migração em um banco produtivo com zero downtime?**

## A Abordagem Tradicional vs. Blue/Green

Manualmente, esse processo é uma dor de cabeça logística. Envolve levantar uma réplica de leitura, promover a atualização, fazer o _downsize_, parar todas as aplicações, trocar os endpoints (DNS) e torcer para nada quebrar ao subir tudo de novo.

Foi aí que optamos pelo **AWS RDS Blue/Green Deployment**.

### O que é o Blue/Green no RDS?

Diferente do deploy de aplicações, onde trocamos containers ou servidores, o Blue/Green no RDS cria um ambiente de "staging" (Green) que é uma cópia exata e sincronizada da produção (Blue).

O serviço replica automaticamente não apenas os dados, mas também conexões, configurações, logs e parâmetros. Isso nos permite aplicar mudanças drásticas no ambiente Green — como upgrades de versão de _engine_ ou alterações de schema — enquanto a produção segue intacta.

## Como funciona o Switchover

A mágica acontece na hora da virada. Após aplicar as mudanças e testar a instância Green , realizamos o _switchover_ via AWS Console (ou CLI/API).

O processo é seguro e praticamente instantâneo, garantindo:

- **Zero downtime perceptível** para o usuário final.

- **Reversibilidade rápida:** se algo der errado no Green, o Blue ainda está lá.

- **Minimização de impacto:** testamos com dados reais, sem afetar a produção.

## Cuidados e "Pegadinhas"

Nem tudo são flores. Antes de adotar essa estratégia, é crucial notar que:

1.  **Custo Temporário:** Durante o processo, você paga pelas duas instâncias (Blue e Green) rodando simultaneamente.

2.  **Incompatibilidades:** Algumas extensões do banco podem não ser compatíveis ou precisam ser desativadas antes do início do processo.

3.  **Engine Suportada:** Nem todas as versões ou engines de banco de dados suportam essa feature.

## Passo a Passo Resumido

### Checklist pré-deploy

- Criar snapshot/manual backup antes de qualquer alteração.
- Verificar compatibilidade de extensões e dependências (ex.: PostGIS, citext).
- Validar que a janela de baixa atividade está definida e comunicar times impactados.
- Checar TTL do DNS e planejar reconexão dos clients / pools (pgbouncer, RDS Proxy).
- Confirmar permissões IAM necessárias para criar/gerenciar Blue/Green deployments.
- Configurar CloudWatch (latência, erros, replication lag) e alarms temporários.
- Calcular custo temporário (duas instâncias rodando) e aprovar orçamento se necessário.
- Preparar scripts de smoke tests e queries de integridade para validação pós-switchover.

Para quem quer aplicar isso hoje, o fluxo no AWS Console é direto:

1. Selecione sua instância atual e vá em **Actions > Create Blue/Green Deployment**.

2. Defina as configurações da nova instância Green (aqui é onde escolhemos a nova versão do motor ou a nova família de instância para reduzir custos).

3. Aguarde a criação e a sincronização. O status ficará como _Available_ para o Blue e _Creating_ para o Green.

4. Realize seus testes no endpoint do Green.

5. Quando estiver seguro, selecione o deployment e clique em **Switch over**.

6. Após validar o sucesso, não esqueça de remover a instância antiga para consolidar a economia.

O uso de Blue/Green transformou uma manutenção crítica e arriscada em um procedimento padrão e seguro, garantindo a performance da aplicação e a saúde financeira da infraestrutura.

---

_Este artigo foi baseado em aprendizados reais sobre migração de banco de dados e otimização de custos em nuvem._
