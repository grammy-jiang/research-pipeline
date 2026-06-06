# Prompt 03 — Blueprint Parse (optional)

You are extracting the product-experience intent from the matching blueprint, if
one is present. The blueprint is **optional but strongly recommended** — it
carries the UX intent the architecture's Experience Architecture was derived
from.

## Inputs

- `intermediate/input_resolution.json` (blueprint path, may be null).
- The blueprint document, if found.

## Instructions

1. If no blueprint is found, record a **warning** ("no blueprint; UX intent taken
   from the architecture's Experience Architecture only") and continue — do not
   stop, provided the architecture has enough information.
2. If a blueprint is found, extract its **§9 Product Experience Direction**:
   primary experience thesis; primary user/operator; job-to-be-done; primary +
   secondary interaction modes (with classification); trust/control/transparency
   requirements; human-in-the-loop experience; failure/recovery expectations; UX
   assumptions for architecture; and the product-experience→architecture handoff.
3. Extract the blueprint's **§19 Recommended Next Stages** routing relevant to
   UX: the `ux-design` decision (RUN/SKIP/DEFER) and any constraints it implies
   (e.g. SKIP → keep UX minimal; RUN → full UX expected).
4. Treat all of this as **UX intent to preserve** — UX design must not change the
   product thesis or UX intent; where it must diverge, record it as architecture
   feedback (prompt 11), not a silent change.

## Output

`intermediate/blueprint_parse.json`:

```json
{
  "blueprint_present": true,
  "experience_thesis": "<...>",
  "primary_user": "<...>",
  "jobs_to_be_done": ["<...>"],
  "interaction_modes": [{"mode": "<...>", "classification": "<...>"}],
  "trust_transparency": ["<...>"],
  "human_in_the_loop": "<...>",
  "failure_recovery_expectations": ["<...>"],
  "ux_design_routing": "RUN | SKIP | DEFER | unknown",
  "warnings": ["<e.g. no blueprint found>"]
}
```

## Validation / failure policy

- Gate: Product Experience Direction extracted, or its absence recorded as a
  warning.
- Failure policy: `warn_continue_if_no_blueprint`.
