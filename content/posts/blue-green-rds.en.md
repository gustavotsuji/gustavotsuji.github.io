```markdown
---
title: 'Zero Downtime and Cloud Savings: How Blue/Green Deployment on RDS Saved Our Budget'
date: '2026-01-26'
excerpt: 'A practical account of how we used Blue/Green Deployment on AWS RDS to migrate and right-size a PostgreSQL with zero downtime, reducing operational costs.'
tags: ['AWS', 'RDS', 'PostgreSQL', 'Blue-Green', 'Database', 'Cloud Cost']
author: 'Gustavo Tsuji'
---

# Zero Downtime and Cloud Savings: How Blue/Green Deployment on RDS Saved Our Budget

TL;DR

- We used Blue/Green Deployment on AWS RDS to migrate and downsize a PostgreSQL with no noticeable downtime.
- Result: estimated cost reduction of $425/month; prerequisites: low-traffic window, snapshots, and integrity tests.
- Pre-requisites: manual snapshot, validate extensions (e.g., PostGIS), check DNS TTL and reconnection strategy for connection pools.

Anyone operating production databases knows the anxiety around critical maintenance. Recently we faced a common but challenging scenario: we needed to upgrade and right-size a mission-critical PostgreSQL that handled hundreds of operations per minute, without impacting user experience.

The situation was this: our database (nicknamed _taffarel_) was overprovisioned. Analysis showed we could cut CPU in half by moving from an `r6g.4xlarge` to an `x2g.2xlarge`, producing a **monthly saving of $425**.

We also faced end-of-life issues for PostgreSQL versions on AWS. Keeping old versions requires extended support (roughly **$413/month**), which is a temporary and costly workaround.

The million-dollar question: **How do you perform this migration on a production database with zero downtime?**

## Traditional Approach vs. Blue/Green

The manual approach is a logistical headache: create a read replica, promote/upgrade, downsize, stop applications, swap endpoints (DNS), and pray nothing breaks when bringing everything back up.

We chose **AWS RDS Blue/Green Deployment** instead.

### What is Blue/Green on RDS?

Unlike application deployments where we swap containers or servers, RDS Blue/Green creates a staging-like environment (Green) that is an exact, synchronized copy of production (Blue).

The service replicates data, connections, settings, logs and parameters automatically. This lets us apply heavy changes — engine upgrades, schema adjustments — on Green while production remains untouched.

## How the Switchover Works

The magic happens at switch time. After applying changes and testing the Green instance, we perform the switchover via the AWS Console (or CLI/API).

The process is safe and near-instant, ensuring:

- **No perceptible downtime** for end users.

- **Quick reversibility:** if something goes wrong on Green, Blue remains.

- **Minimal impact:** we test with real data without affecting production.

## Caveats and Gotchas

Not everything is perfect. Before adopting this strategy, note that:

1. **Temporary cost:** you pay for both Blue and Green while synchronization occurs.

2. **Incompatibilities:** some database extensions may be incompatible or must be disabled before starting.

3. **Engine support:** not all versions or database engines support this feature.

## Pre-Deploy Checklist

- Take a manual snapshot / backup before any change.
- Verify extension and dependency compatibility (e.g., PostGIS, citext).
- Ensure you have a low-traffic window scheduled and notify impacted teams.
- Check DNS TTL and plan client/pool reconnection (pgbouncer, RDS Proxy).
- Confirm IAM permissions required to create/manage Blue/Green deployments.
- Configure CloudWatch (latency, errors, replication lag) and temporary alarms.
- Calculate temporary cost (two instances running) and approve budget if needed.
- Prepare smoke test scripts and integrity queries for post-switchover validation.

## Step-by-step Summary

For those who want to apply this today, the console flow is straightforward:

1. Select your current instance and choose **Actions > Create Blue/Green Deployment**.

2. Define the Green instance configuration (where you choose the engine version or new instance family to reduce cost).

3. Wait for creation and synchronization. The status will show _Available_ for Blue and _Creating_ for Green.

4. Run tests against the Green endpoint.

5. When ready, select the deployment and click **Switch over**.

6. After validating success, remove the old instance to consolidate savings.

Using Blue/Green turned a risky, critical maintenance into a repeatable, safe procedure that preserved application performance and improved infrastructure cost health.

---

_This article is based on real learnings about database migrations and cloud cost optimization._
```
