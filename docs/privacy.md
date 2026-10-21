# Privacy and data protection

Mitate is designed for India's Digital Personal Data Protection Act 2023 and the DPDP Rules 2025
(G.S.R. 846(E), 13 Nov 2025). The legal analysis, with quoted sections and sources, is in
[research_notes/06_dpdp.md](research_notes/06_dpdp.md). This page states the engineering rules that
follow from it. It is not legal advice; merchants should have counsel review the data processing
agreement and their notices.

## Timeline that matters

| Date | What starts |
|---|---|
| 13 Nov 2025 | Definitions, Data Protection Board |
| ≈ 13 Nov 2026 | Consent Manager registration (Rule 4) |
| ≈ 13 May 2027 | All substantive duties: notice, consent, security, breach notification, retention, children, rights |

A proposal to compress the 18 months to 12 was reported in early 2026; no amending notification
was found as of October 2026. Mitate targets being compliant by **13 Nov 2026** as a hedge.

## Roles

- The **merchant is the Data Fiduciary**: it decides to personalise and owns the shopper relationship.
- **Mitate is a Data Processor** when it processes one merchant's events to serve that merchant's
  recommendations under contract (s.8(2)).
- Mitate would become a Fiduciary for any purpose it decides itself, such as pooling shoppers'
  events across merchants. So: **per-merchant isolation is the default**, and cross-merchant
  learning is limited to aggregated, non-personal signals (for example a category-level
  complement graph or the blend's ranking model trained on public benchmark data), unless the
  merchant opts in and its shopper notice lists that purpose.

## Engineering rules

| # | Rule | Source | Where it lives |
|---|---|---|---|
| P1 | **Data minimisation.** Events carry only: pseudonymous shopper id, session id, item id, event type, timestamp, and optional coarse context (channel, device class, region/PIN prefix for delivery). No name, email, phone, address, precise location, or free text. | s.6(1), s.8(7) | `serve/schemas.py`: every inbound model has `extra="forbid"`; test `test_personal_fields_are_rejected` |
| P2 | **No sensitive attributes, inferred or supplied.** Gender, age, religion, caste and similar are never ingested, inferred or used as features. Age only appears as the merchant-supplied `age_status` flag that *restricts* processing (P5). | Build prompt; s.9 | no schema field can carry them; the blend's features (`mitate/blend.py`) are behavioural, catalog and store-volume only |
| P3 | **Pseudonymous ids.** Merchants send their own opaque id; Mitate stores `HMAC-SHA256(merchant_secret, id)` and never the raw value. Pseudonymous data is still personal data (s.2(t)), so P4–P8 still apply. | Rule 6(1)(a) | `serve/privacy.py` (`pseudonym`), per-merchant 32-byte secret; test `test_only_consented_events_are_stored_and_ids_are_pseudonymised` greps the database file for raw ids |
| P4 | **Consent state on every event.** `consent ∈ {granted, denied, withdrawn, unknown}`. Without `granted`, Mitate writes no profile and serves the non-personalised path: popularity and item-to-item from the item on screen only. Withdrawal stops processing immediately and triggers P6. | s.4, s.6(4), s.6(6) | `serve/privacy.py` `Gate`; `Service.ingest` drops events without consent; `Service.recommend` uses only the item on screen; the widget stores no id without consent |
| P5 | **Minor-safe path.** `age_status ∈ {adult_verified, adult_declared, minor, unknown}`. For `minor`: no profile, no session-behaviour modelling, no targeted promotions; only contextual non-personalised recommendations. Whether in-visit personalisation counts as "behavioural monitoring" is unclear, so it is treated as prohibited. Merchants in child-heavy categories should map `unknown` to the minor-safe path. | s.9(3) (no e-commerce exemption; penalty up to ₹200 crore) | `Gate.of`; rule `minor_safe_for_unknown_age`; test `test_minor_and_unconsented_requests_are_not_personalised` |
| P6 | **Erasure and export.** `DELETE /shoppers/{id}` removes the shopper from serving features and the training store and adds the HMAC to a suppression list; the next scheduled retrain excludes them. `GET /shoppers/{id}/export` returns what is held. | s.8(7), s.11–12, Rule 14 | `Service.erase` / `Service.export`; erasure also removes the shopper's sessions and experiment exposure, adds a suppression entry and triggers a retrain at once; test `test_erasure_removes_data_and_suppresses_future_events` checks the retrained model excludes them |
| P7 | **Retention.** Behavioural features have a merchant-configurable TTL (default 180 days). Separately, a restricted, append-only processing log (who/what/when of processing, no event payloads) is kept for at least one year, as Rule 8(3) requires, and then purged. | s.8(7), Rule 8(3), Rule 6(1)(e) | `Service.purge` (run at every retrain), rule `retention_days`; `processing_log` holds action and count only, kept 400 days |
| P8 | **No raw personal data in logs.** Application logs contain request ids, timings and counts only; shopper ids appear only as HMACs. | Rule 6 | `serve/app.py` access middleware logs route templates and timings only; uvicorn's URL access log is off; shopper ids travel in POST bodies, never URLs; test `test_logs_never_contain_ids` |
| P9 | **Security baseline.** Encryption at rest is the deployer's responsibility (documented); API keys per merchant; access logging. | Rule 6(1) | deployment guide |
| P10 | **Breach support.** Mitate commits to notifying the merchant within 24 hours with the Rule 7(2) fields, so the merchant can meet its 72-hour report to the Board. | Rule 7 | DPA template (to be written with counsel) |

## The benchmark

The public benchmark uses non-Indian, already-anonymised datasets (OTTO, H&M, Dunnhumby, Amazon),
which are very likely not personal data under s.2(t). No merchant's real data is used in any
reported number. The demo store is synthetic (Misata) and labelled as such.

## Status

P1–P8 are implemented in `serve/` and covered by `tests/test_serve.py`. P9 and P10 depend on the
deployment and the merchant's contract and are not things code can guarantee. Implementing these
rules is not a compliance certification: the merchant remains the Data Fiduciary and needs its
own notice, consent flow and counsel review.

What the code does **not** do yet: verify ages (it trusts the merchant's `age_status`), run as a
registered Consent Manager, or encrypt the SQLite file (deployers should put it on an encrypted
volume, P9).
