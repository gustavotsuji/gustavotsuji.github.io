---
title: 'PostgreSQL Table Partitioning: Performance at Scale'
date: '2024-01-10'
excerpt: 'Advanced partitioning strategies to handle billions of records efficiently. Real-world examples and performance benchmarks from production systems.'
tags: ['PostgreSQL', 'Database', 'Performance', 'Scalability']
author: 'Gustavo Tsuji'
---

# PostgreSQL Table Partitioning: Performance at Scale

When your PostgreSQL tables grow to millions or billions of rows, query performance can degrade significantly. Table partitioning is a powerful technique to maintain performance at scale. Here's what I learned implementing it in production.

## Why Partition?

Without partitioning, a query on a 100M row table might scan the entire table. With partitioning, PostgreSQL can skip entire partitions, dramatically improving performance.

### Real-World Example

In a high-traffic marketplace, we had a `product_listings` table with:

- **50 million rows**
- **Growing by 100k rows/day**
- **Queries taking 15+ seconds**

After partitioning by date: **Queries dropped to <500ms** ⚡

## Partition Strategies

### 1. Range Partitioning (Time-Based)

Perfect for time-series data.

```sql
-- Create parent table
CREATE TABLE product_listings (
    id BIGSERIAL,
    listing_date DATE NOT NULL,
    product_id INTEGER NOT NULL,
    price DECIMAL(10,2),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (listing_date);

-- Create partitions for each month
CREATE TABLE product_listings_2024_01 PARTITION OF product_listings
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE product_listings_2024_02 PARTITION OF product_listings
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE product_listings_2024_03 PARTITION OF product_listings
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
```

### 2. List Partitioning (Category-Based)

Perfect for categorical data.

```sql
CREATE TABLE orders (
    id BIGSERIAL,
    order_date DATE,
    status VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2)
) PARTITION BY LIST (status);

CREATE TABLE orders_pending PARTITION OF orders
    FOR VALUES IN ('pending', 'processing');

CREATE TABLE orders_completed PARTITION OF orders
    FOR VALUES IN ('completed', 'shipped');

CREATE TABLE orders_cancelled PARTITION OF orders
    FOR VALUES IN ('cancelled', 'refunded');
```

### 3. Hash Partitioning (Load Distribution)

Perfect for distributing data evenly.

```sql
CREATE TABLE user_events (
    id BIGSERIAL,
    user_id INTEGER NOT NULL,
    event_type VARCHAR(50),
    event_data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
) PARTITION BY HASH (user_id);

CREATE TABLE user_events_p0 PARTITION OF user_events
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);

CREATE TABLE user_events_p1 PARTITION OF user_events
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);

CREATE TABLE user_events_p2 PARTITION OF user_events
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);

CREATE TABLE user_events_p3 PARTITION OF user_events
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

## Automatic Partition Management

Manual partition creation doesn't scale. Here's how to automate it:

```sql
-- Function to create monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partitions(
    table_name TEXT,
    start_date DATE,
    end_date DATE
)
RETURNS void AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_range DATE;
    end_range DATE;
BEGIN
    partition_date := DATE_TRUNC('month', start_date);

    WHILE partition_date < end_date LOOP
        partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
        start_range := partition_date;
        end_range := partition_date + INTERVAL '1 month';

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
             FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            table_name,
            start_range,
            end_range
        );

        partition_date := partition_date + INTERVAL '1 month';
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Create partitions for the next 12 months
SELECT create_monthly_partitions(
    'product_listings',
    CURRENT_DATE,
    CURRENT_DATE + INTERVAL '12 months'
);
```

## Indexes on Partitioned Tables

Each partition needs its own indexes:

```sql
-- Create index on parent (applies to all partitions)
CREATE INDEX idx_product_listings_product_id
    ON product_listings(product_id);

CREATE INDEX idx_product_listings_status
    ON product_listings(status)
    WHERE status = 'active';

-- Verify indexes were created on all partitions
SELECT
    schemaname,
    tablename,
    indexname
FROM pg_indexes
WHERE tablename LIKE 'product_listings%'
ORDER BY tablename, indexname;
```

## Querying Partitioned Tables

PostgreSQL automatically uses partition pruning:

```sql
-- This query only scans January partition
EXPLAIN ANALYZE
SELECT * FROM product_listings
WHERE listing_date BETWEEN '2024-01-01' AND '2024-01-31';

/*
Result:
  Seq Scan on product_listings_2024_01
  (cost=0.00..1234.56 rows=50000 width=100)
  Planning Time: 0.5ms
  Execution Time: 45ms
*/

-- Without partitioning, would scan ALL 50M rows!
```

## Migration Strategy

You can't partition an existing table directly. Here's the migration process:

```sql
-- Step 1: Rename existing table
ALTER TABLE product_listings RENAME TO product_listings_old;

-- Step 2: Create partitioned table
CREATE TABLE product_listings (
    LIKE product_listings_old INCLUDING ALL
) PARTITION BY RANGE (listing_date);

-- Step 3: Create partitions
SELECT create_monthly_partitions(
    'product_listings',
    '2020-01-01'::DATE,
    CURRENT_DATE + INTERVAL '3 months'
);

-- Step 4: Copy data in batches
DO $$
DECLARE
    batch_size INTEGER := 100000;
    offset_val INTEGER := 0;
    rows_copied INTEGER;
BEGIN
    LOOP
        INSERT INTO product_listings
        SELECT * FROM product_listings_old
        ORDER BY id
        LIMIT batch_size OFFSET offset_val;

        GET DIAGNOSTICS rows_copied = ROW_COUNT;
        EXIT WHEN rows_copied = 0;

        offset_val := offset_val + batch_size;
        RAISE NOTICE 'Copied % rows', offset_val;

        -- Commit every batch
        COMMIT;
    END LOOP;
END $$;

-- Step 5: Verify data
SELECT COUNT(*) FROM product_listings;
SELECT COUNT(*) FROM product_listings_old;

-- Step 6: Drop old table (after verification!)
-- DROP TABLE product_listings_old;
```

## Performance Comparison

Real production benchmarks from a high-traffic marketplace:

| Query Type           | Without Partitioning | With Partitioning | Improvement    |
| -------------------- | -------------------- | ----------------- | -------------- |
| Date range (1 month) | 15.2s                | 0.4s              | **38x faster** |
| Date range (1 week)  | 8.7s                 | 0.15s             | **58x faster** |
| Single day           | 3.2s                 | 0.08s             | **40x faster** |
| Count by status      | 25.3s                | 1.2s              | **21x faster** |

## Maintenance Tasks

### Detach Old Partitions

```sql
-- Detach partition (keeps data)
ALTER TABLE product_listings
    DETACH PARTITION product_listings_2020_01;

-- Archive old data
CREATE TABLE archived_listings_2020_01 AS
SELECT * FROM product_listings_2020_01;

-- Drop old partition
DROP TABLE product_listings_2020_01;
```

### Vacuum Strategy

```sql
-- Vacuum each partition separately
DO $$
DECLARE
    partition_name TEXT;
BEGIN
    FOR partition_name IN
        SELECT tablename
        FROM pg_tables
        WHERE tablename LIKE 'product_listings_%'
        ORDER BY tablename
    LOOP
        EXECUTE 'VACUUM ANALYZE ' || partition_name;
        RAISE NOTICE 'Vacuumed %', partition_name;
    END LOOP;
END $$;
```

## Common Pitfalls

1. **Wrong partition key** - Choose a column frequently used in WHERE clauses
2. **Too many partitions** - Keep it under 100-200 partitions
3. **Missing indexes** - Create indexes on each partition
4. **No automation** - Automate partition creation
5. **Ignoring constraints** - Add constraints for better query optimization

## Monitoring

```sql
-- Check partition sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    pg_total_relation_size(schemaname||'.'||tablename) AS bytes
FROM pg_tables
WHERE tablename LIKE 'product_listings%'
ORDER BY bytes DESC;

-- Check partition pruning
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM product_listings
WHERE listing_date = '2024-01-15';
```

## Conclusion

Table partitioning transformed our database performance:

- **38x faster** queries on average
- **Simpler** data lifecycle management
- **Better** maintenance windows
- **Scalable** to billions of rows

Start partitioning when:

- Tables exceed 10M rows
- Queries are slowing down
- You have a natural partition key (date, region, status)

---

_Questions about PostgreSQL partitioning? Let's discuss on [LinkedIn](https://linkedin.com/in/gustavo-tsuji-7100462b)!_
