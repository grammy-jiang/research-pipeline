# Daily briefing replay

Daily AI intelligence runs write a local `workflow_state.json` under:

```text
workspace/briefings/YYYY-MM-DD/workflow_state.json
```

Use it to inspect completed stages and resume from existing artifacts:

```bash
research-pipeline brief resume --date YYYY-MM-DD --from-stage rank
research-pipeline brief resume --date YYYY-MM-DD --from-stage generate-daily
research-pipeline brief resume --date YYYY-MM-DD --from-stage validate
```

Replay rules:

- `rank` reads `normalized/events.jsonl` and rebuilds clusters, ranked clusters,
  daily report, and validation.
- `generate-daily` reads `ranked/ranked_clusters.jsonl` and rebuilds the report
  and validation.
- `validate` reads the existing report and ranked clusters.

Source expansion should be compared before enablement:

```bash
research-pipeline brief compare-sources \
  --base-registry base.json \
  --expanded-registry expanded.json \
  --date YYYY-MM-DD
```
