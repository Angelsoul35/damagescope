# ADR-0001: Record architecture decisions

* Status: accepted
* Date: 2026-02-24

## Context

DamageScope makes a few choices that are not obvious (classical CV over a deep model,
residual-based localization, ONNX export). A future reader needs the *why*.

## Decision

We keep short Architecture Decision Records in `docs/adr/`, one per important choice, each
with Context, Decision, and Consequences.

## Consequences

* Choices are easy to review later.
* The cost is one short Markdown file per decision.
