# Feedback loop

Use explicit feedback first:

```bash
research-pipeline brief feedback --cluster <cluster_id> --signal keep
research-pipeline brief feedback --topic <topic_id> --signal too_noisy
research-pipeline brief feedback --source <source_id> --signal less_like_this
```

Feedback is stored locally and affects future deterministic ranking. Behavioral
signals should not be promoted into durable ranking without review.
