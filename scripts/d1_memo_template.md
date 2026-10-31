{#- Rendered by scripts/run_d1.py with jinja2 (StrictUndefined, trim_blocks, lstrip_blocks). run_d1 builds every
    row and sanitises every cell; this file holds the order and the words. ASCII only. -#}
# D1 decision memo{{ " (DRY RUN)" if dry else "" }}

{% if dry %}
**DRY RUN.** Every number below comes from the 10/26 probe fixtures (the cookbook's ping pong clip, canned Spark and
Voice answers), not from a car. It shows that the gates, the runner and this memo work end to end. It decides
nothing, and a pass or fail here only says a code path ran.

{% endif %}
Run `{{ meta.run_prefix }}`, started {{ meta.started_at }}, finished {{ meta.finished_at }}, folder {{ meta.out }}.
Spend on ledger rows with this prefix: ${{ meta.spend }} of the ${{ meta.budget }} ceiling{% if dry %} (dry-run ledgers, nothing spent){% else %}; the whole ledger stands at ${{ meta.total_ledger }}{% endif %}.
Thresholds config hash {{ meta.config_hash }}. Plan: {{ plan }}.

Criteria: {{ counts["pass"] }} pass, {{ counts["fail"] }} fail, {{ counts["not run"] }} not run, {{ counts["info"] }} measured with no bar.

## Recommendation

**{{ recommendation.headline }}**

{% for line in recommendation.lines %}
- {{ line }}
{% endfor %}

## Decisions

The list section 11 asks the memo for, each with the criteria it comes from.

| Item | Decision | From |
|---|---|---|
{% for d in decisions %}
| {{ d.item }} | {{ d.value }} | {{ d.source }} |
{% endfor %}

## What failed, and section 11's branch for it

{% if fails %}
| Criterion | Measured | Section 11's branch |
|---|---|---|
{% for c in fails %}
| {{ c.id }} | {{ c.value }} | {{ c.on_fail }} |
{% endfor %}
{% else %}
Nothing failed{% if counts["not run"] %}; {{ counts["not run"] }} criteria did not run, each with its reason below{% endif %}.
{% endif %}

## Inputs

{% if inputs %}
| Role | File | sha256, first 16 | Camera time | Video |
|---|---|---|---|---|
{% for i in inputs %}
| {{ i.role }} | {{ i.path }} | {{ i.sha256 }} | {{ i.camera_time }} | {{ i.size }} |
{% endfor %}
{% else %}
E0 did not run, so no original was hashed.
{% endif %}
{% if picks %}

Picked by the runner where the plan named nothing:

{% for p in picks %}
- {{ p }}
{% endfor %}
{% endif %}

## Every criterion, by row

{% for row in rows %}
### {{ row.id }} {{ row.title }} ({{ row.cost }}): {{ row.status }}

Step `{{ row.step }}`: {{ row.step_status }}{% if row.reason %} ({{ row.reason }}){% endif %}.

| Criterion | Rule, as section 11 writes it | Measured | Result | Note |
|---|---|---|---|---|
{% for c in row.criteria %}
| {{ c.id }} | {{ c.rule }} | {{ c.value }} | {{ c.status }} | {{ c.note }} |
{% endfor %}

{% endfor %}
## E13 dollars from the parts

{{ e13_note }}

{% if e13_pair %}
| Pair item | Count | Unit $ | $ | Basis |
|---|---|---|---|---|
{% for i in e13_pair %}
| {{ i.item }} | {{ i.count }} | {{ i.unit_usd }} | {{ i.usd }} | {{ i.basis }} |
{% endfor %}

| Pickup-only item | Count | Unit $ | $ | Basis |
|---|---|---|---|---|
{% for i in e13_pickup %}
| {{ i.item }} | {{ i.count }} | {{ i.unit_usd }} | {{ i.usd }} | {{ i.basis }} |
{% endfor %}
{% else %}
E13 did not run.
{% endif %}

## Steps and spend

| Step | Rows | Status | Seconds | Planned $ | Spent $ | Reason |
|---|---|---|---|---|---|---|
{% for s in spend_rows %}
| {{ s.step }} | {{ s.rows }} | {{ s.status }} | {{ s.seconds }} | {{ s.planned }} | {{ s.spent }} | {{ s.reason }} |
{% endfor %}

## Files

{% for f in files %}
- {{ f }}
{% endfor %}
- results.json holds every number above with the gate JSONs it came from.
{% if deleted %}

Unblurred intermediates deleted at the end of the run:

{% for d in deleted %}
- {{ d }}
{% endfor %}
{% endif %}

Next: copy this memo to docs/D1_MEMO.md in the private workspace; Codex reads it and writes codex/SUGGESTIONS.md, and the
README's measured rows are drafted from it, each number with the script and the run that produced it.
