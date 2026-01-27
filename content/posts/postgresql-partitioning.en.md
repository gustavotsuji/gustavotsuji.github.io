---
title: 'PostgreSQL Table Partitioning: Performance at Scale'
date: '2024-01-10'
excerpt: 'Advanced partitioning strategies to handle billions of records efficiently. Real-world examples and performance benchmarks from production systems.'
tags: ['PostgreSQL', 'Database', 'Performance', 'Scalability']
author: 'Gustavo Tsuji'
---

# Scaling PostgreSQL: How Table Partitioning Solves Performance Bottlenecks

Anyone working with large data volumes knows that, eventually, table size starts taking a toll. Queries become slow, indexes grow to massive sizes, and maintenance operations can overload the database.

Recently, we faced performance and cost issues caused by very large tables, where high VACUUM resource consumption on giant tables led to extreme latency and timeouts in critical processes.

The solution adopted to mitigate this was **Table Partitioning**, which divides these tables into smaller parts to optimize resources. In this article, I explore what this technique is, how it works in PostgreSQL, and when you should (or should not) use it. I will also explain how to use other resources (`pg_cron` and `pg_partman`) to automate partitioning, as well as the deletion of old data.

## What is Table Partitioning?

Partitioning is a technique that consists of splitting a large table into smaller parts, called "partitions".

The magic happens in its transparency: for the application and the end-user, these parts still behave as if they were a single table. Physically, however, each partition stores only a subset of the data (defined by criteria such as date, status, or ID).

PostgreSQL assumes the responsibility of automatically deciding which partition to store or retrieve data from based on the rules you define.

## Why use it?

Beyond resolving the problem of giant tables, partitioning brings important structural benefits:

1.  **Read Performance:** Significant improvement in large tables, especially for reads and filters that utilize the partition column.
2.  **Maintenance and Cleanup:** Imagine having to delete millions of old rows. With partitioning, archiving or cleaning up old data is facilitated, allowing for the `DROP` of partitions instead of costly `DELETE`s.
3.  **Optimized Indexes:** Instead of a giant monolithic index, you have smaller, more specific indexes, which improves maintenance.
4.  **Resource Usage:** Improves disk space usage and makes `autovacuum` execution much more efficient.

## Partitioning Strategies

PostgreSQL offers three main native strategies:

- **RANGE:** Ideal when data is organized by intervals, such as dates. _Ex: Event table partitioned by year/month._
- **LIST:** Used when data has fixed categories, such as status or region.
- **HASH:** When you want to distribute data in a balanced way, without a specific logical order.

## How it works in practice

Implementation in PostgreSQL is declarative. Below is an example of how we create a table partitioned by date (RANGE) to store events:

```sql
-- 1. Create the partitioned master table
CREATE TABLE events (
    id SERIAL,
    event_date DATE NOT NULL,
    description TEXT,
    PRIMARY KEY (id, event_date)
) PARTITION BY RANGE (event_date);

-- 2. Create specific partitions
CREATE TABLE events_2023 PARTITION OF events
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE events_2024 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

Aqui está o markdown da parte solicitada (do trecho "When executing a query..." até o final):

Markdown

When executing a query like `SELECT * FROM events WHERE event_date >= '2024-01-01';`, the database performs automatic _pruning_ and only accesses the relevant partitions.

## Common Use Cases

Partitioning is ideal in scenarios such as:

- Logs organized by date;
- Historical event data;
- Fiscal files, invoices, and transactions;
- Jobs, notifications, and status by category.

## It's not all sunshine and rainbows: Cautions and when NOT to use

Partitioning requires planning. There are operational complexities that need to be considered:

1.  **Manual Management:** Partitions generally need to be created manually or via tools like `pg_partman`.
2.  **Indexes and Keys:** Although modern versions of PostgreSQL propagate indexes automatically, the definition of **Primary Keys** and **Unique Constraints** requires the inclusion of the partition key, which impacts data modeling.
3.  **Restrictions:** There are restrictions for _Triggers_, _Constraints_, and _Foreign Keys_ on partitioned tables.
4.  **Join Performance:** Operations involving many partitions can be slower if not well-planned.

**Mainly, avoid partitioning if:**

- Your table is small or medium-sized (it won't bring real gain).
- Read volume is high, but the query filter does **not** use the partition column.
- You don't have a planned maintenance strategy (like DROP routines or archiving).

## How to automate partitioning?

Creating partitions manually is an operational risk. Forgetting to create next month's partition, for example, can generate complications. There is a "fallback" partition where, in case Postgres doesn't find the destination partition, any data ends up being directed to this partition (usually defined as default).

To solve this robustly, the standard combination is to use two extensions:

1.  **pg_partman:** Manages the automatic creation of new partitions and the disposal of old ones.
2.  **pg_cron:** A job scheduler that runs _inside_ the database, eliminating the need for complex scripts on the operating system.

### Step 1: Enabling extensions

First, ensure that the extensions are installed and enabled in your database (`pg_cron` usually requires prior configuration in `shared_preload_libraries` within `postgresql.conf`).

```sql
CREATE EXTENSION IF NOT EXISTS pg_partman WITH SCHEMA partman;
CREATE EXTENSION IF NOT EXISTS pg_cron;
```

With this, `pg_partman` understands the structure, but it doesn't run by itself. It needs to be "called" periodically to check if it's time to create new tables.

### Step 3: Scheduling Maintenance (pg_cron)

This is where `pg_cron` shines. Instead of configuring a `crontab` in Linux, we schedule the execution of the maintenance routine directly via SQL.

The command below configures the database to run `partman` maintenance every hour:

```sql
-- Schedule the execution of run_maintenance() every hour
SELECT cron.schedule(
    'partition_maintenance', -- Job name
    '0 * * * *',             -- Cron expression (every hour, minute 0)
    $$CALL partman.run_maintenance()$$
);
```

### The Result

With this configuration:

1.  Every hour, `pg_cron` wakes up and executes `run_maintenance()`.
2.  `pg_partman` checks if future partitions (defined in `p_premake`) already exist.
3.  If they don't exist, they are automatically created, ensuring your application never fails due to a lack of tables to insert new data.
4.  (Optional) If configured, it can also automatically detach or archive very old partitions.

This architecture transforms partitioning from a "maintenance headache" into an "invisible performance solution".
