# Forbidden Content Checklist

The blueprint MUST NOT contain any of the following. If a reviewer finds
one, the blueprint fails Gate 3 (Implementation Neutrality) and must be
revised.

## Tech-stack selections (forbidden)

- [ ] Programming language choice (e.g. "use Python / TypeScript / Go").
- [ ] Framework choice (e.g. FastAPI, Django, React, Next.js, Spring).
- [ ] Database choice, including specific products (PostgreSQL, MySQL,
      MongoDB, Redis, SQLite, DynamoDB).
- [ ] Vector database / library (Chroma, FAISS, Pinecone, Milvus, Weaviate).
- [ ] Cloud provider (AWS, GCP, Azure) or vendor-specific service.
- [ ] UI library or component kit.
- [ ] Message broker / queue product (Kafka, RabbitMQ, SQS).

## Implementation artifacts (forbidden)

- [ ] Source code in any language.
- [ ] Database schema / `CREATE TABLE` / migration statements.
- [ ] Class, module, or package definitions.
- [ ] Concrete package/module/repository structure.
- [ ] Deployment commands, Dockerfiles, compose files, IaC.
- [ ] API endpoint definitions (REST/gRPC routes, request/response bodies).
- [ ] Implementation tickets, sprint tasks, or detailed coding tasks.
- [ ] `pip install` / `npm install` / dependency lists.

## Reasoning errors (forbidden)

- [ ] An open `ACADEMIC` research gap silently treated as a solved
      feature.
- [ ] `ACADEMIC`-gap item placed in MVP without explicit validation
      justification.
- [ ] A HIGH-impact risk omitted or softened.
- [ ] "Prompt the model better" offered as a sufficient mitigation.
- [ ] Invented performance numbers / benchmark figures not in the source
      report.

## Allowed (do not flag)

Conceptual responsibilities and capabilities — e.g. "durable record
store", "append-only event ledger", "vector similarity search", "process
isolation between tenants", and neutral component names like "Admission
Controller". See `references/borderline-cases.md`.
