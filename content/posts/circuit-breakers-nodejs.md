---
title: 'Building Resilient Microservices with Circuit Breakers'
date: '2024-02-20'
excerpt: 'Implementing fault tolerance patterns in Node.js applications for production resilience. Learn how to handle failures gracefully and maintain system stability.'
tags: ['Node.js', 'Architecture', 'Resilience', 'Microservices']
author: 'Gustavo Tsuji'
---

# Building Resilient Microservices with Circuit Breakers

In distributed systems, failures are inevitable. The question isn't _if_ a service will fail, but _when_. This article explores how to use the Circuit Breaker pattern to build resilient microservices in Node.js.

## The Problem

Imagine this scenario:

- Service A calls Service B
- Service B is experiencing high latency (10+ seconds per request)
- Service A keeps waiting, tying up resources
- Eventually, Service A crashes from resource exhaustion

This is called a **cascading failure**.

## The Circuit Breaker Pattern

A circuit breaker prevents cascading failures by "breaking the circuit" when a service is unhealthy.

### States

1. **CLOSED** - Normal operation, requests flow through
2. **OPEN** - Service is unhealthy, requests fail fast
3. **HALF_OPEN** - Testing if service recovered

```
CLOSED ---[threshold exceeded]---> OPEN
  ^                                  |
  |                                  |
  +---[success]--- HALF_OPEN <-------+
                      |
                      +---[failure]---> OPEN
```

## Implementation in Node.js

Let's implement a simple circuit breaker:

```typescript
class CircuitBreaker {
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED'
  private failureCount = 0
  private successCount = 0
  private nextAttempt = Date.now()

  constructor(
    private threshold = 5, // failures to open circuit
    private timeout = 60000, // ms to wait before retry
    private monitoringPeriod = 10000 // ms to reset counter
  ) {}

  async call<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN')
      }
      this.state = 'HALF_OPEN'
    }

    try {
      const result = await fn()
      this.onSuccess()
      return result
    } catch (error) {
      this.onFailure()
      throw error
    }
  }

  private onSuccess() {
    this.failureCount = 0

    if (this.state === 'HALF_OPEN') {
      this.successCount++
      if (this.successCount >= 2) {
        this.state = 'CLOSED'
        this.successCount = 0
      }
    }
  }

  private onFailure() {
    this.failureCount++
    this.successCount = 0

    if (this.failureCount >= this.threshold) {
      this.state = 'OPEN'
      this.nextAttempt = Date.now() + this.timeout
    }
  }

  getState() {
    return this.state
  }
}
```

## Real-World Usage

Here's how we use it at OLX:

```typescript
import axios from 'axios'

// Create circuit breaker for external API
const paymentServiceBreaker = new CircuitBreaker(
  5, // open after 5 failures
  30000, // wait 30s before retry
  10000 // reset counter every 10s
)

async function processPayment(orderId: string) {
  try {
    const response = await paymentServiceBreaker.call(async () => {
      return await axios.post('https://payment-service/api/process', {
        orderId,
        timeout: 5000,
      })
    })

    return response.data
  } catch (error) {
    if (error.message === 'Circuit breaker is OPEN') {
      // Fallback: queue for later processing
      await queuePayment(orderId)
      return { status: 'queued' }
    }
    throw error
  }
}
```

## Advanced Features

### 1. Metrics & Monitoring

```typescript
class MonitoredCircuitBreaker extends CircuitBreaker {
  private metrics = {
    totalCalls: 0,
    successfulCalls: 0,
    failedCalls: 0,
    rejectedCalls: 0,
  }

  async call<T>(fn: () => Promise<T>): Promise<T> {
    this.metrics.totalCalls++

    if (this.state === 'OPEN') {
      this.metrics.rejectedCalls++
      throw new Error('Circuit breaker is OPEN')
    }

    try {
      const result = await super.call(fn)
      this.metrics.successfulCalls++
      return result
    } catch (error) {
      this.metrics.failedCalls++
      throw error
    }
  }

  getMetrics() {
    return { ...this.metrics, state: this.state }
  }
}
```

### 2. Integration with Express

```typescript
import express from 'express'

const app = express()
const apiBreaker = new MonitoredCircuitBreaker()

app.get('/api/data', async (req, res) => {
  try {
    const data = await apiBreaker.call(async () => {
      return await fetchDataFromExternalAPI()
    })
    res.json(data)
  } catch (error) {
    if (error.message === 'Circuit breaker is OPEN') {
      res.status(503).json({
        error: 'Service temporarily unavailable',
        retryAfter: 30,
      })
    } else {
      res.status(500).json({ error: 'Internal server error' })
    }
  }
})

// Metrics endpoint
app.get('/metrics/circuit-breaker', (req, res) => {
  res.json(apiBreaker.getMetrics())
})
```

## Best Practices

1. **Set appropriate thresholds** - Too sensitive = false positives, too lenient = cascading failures
2. **Implement fallbacks** - What happens when circuit opens?
3. **Monitor metrics** - Track state changes and failure rates
4. **Use different breakers** - One per external dependency
5. **Consider timeouts** - Always set request timeouts

## Testing

```typescript
describe('CircuitBreaker', () => {
  it('should open circuit after threshold failures', async () => {
    const breaker = new CircuitBreaker(3, 1000)
    const failingFn = jest.fn().mockRejectedValue(new Error('fail'))

    // Trigger 3 failures
    for (let i = 0; i < 3; i++) {
      try {
        await breaker.call(failingFn)
      } catch (e) {}
    }

    expect(breaker.getState()).toBe('OPEN')
  })

  it('should transition to HALF_OPEN after timeout', async () => {
    const breaker = new CircuitBreaker(2, 100)

    // Open the circuit
    for (let i = 0; i < 2; i++) {
      try {
        await breaker.call(async () => {
          throw new Error()
        })
      } catch (e) {}
    }

    expect(breaker.getState()).toBe('OPEN')

    // Wait for timeout
    await new Promise((resolve) => setTimeout(resolve, 150))

    // Next call should attempt (HALF_OPEN)
    try {
      await breaker.call(async () => 'success')
    } catch (e) {}

    expect(breaker.getState()).toBe('CLOSED')
  })
})
```

## Libraries

For production use, consider these battle-tested libraries:

- **[opossum](https://nodeshift.dev/opossum/)** - Full-featured circuit breaker
- **[cockatiel](https://github.com/connor4312/cockatiel)** - Resilience patterns library
- **[brakes](https://github.com/awolden/brakes)** - Hystrix-style circuit breaker

## Conclusion

Circuit breakers are essential for building resilient microservices. They:

- Prevent cascading failures
- Provide fast failure feedback
- Allow systems to recover gracefully
- Improve overall system stability

Start small, monitor closely, and adjust thresholds based on real-world behavior.

---

_Have you implemented circuit breakers in your services? Share your experience on [LinkedIn](https://linkedin.com/in/gustavo-tsuji-7100462b)!_
