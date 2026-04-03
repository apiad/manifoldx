---
description: Review prose, technical writing and docs for structural, content, and style.
name: editor
---

This skill provides structured, pragmatic review instructions across four dimensions. Always read the full document once first, then apply the relevant review checklist.

**Review order:** Structure → Content → Style → General

---

## Subsections

### Structural Review
- **File:** `subsections/structural-review.md`
- **Purpose:** Analyze narrative arc, logical flow, organization, transitions, paragraph structure, how main points connect
- **When to use:** Large documents, persuasive pieces, technical articles, long-form content

### Content Review
- **File:** `subsections/content-review.md`
- **Purpose:** Analyze factuality, methodology, argument strength, evidence quality, completeness
- **When to use:** Research papers, technical documentation, opinion pieces, anything making claims

### Style Review
- **File:** `subsections/style-review.md`
- **Purpose:** Analyze grammar, wording, spelling, tone, voice consistency, clarity, readability
- **When to use:** Drafts ready for publication, polished prose, any document needing line-level scrutiny

### General Review
- **File:** `subsections/general-review.md`
- **Purpose:** Form overall impression, assess cohesion, evaluate whether the piece achieves its purpose
- **When to use:** Final review pass, holistic assessment, reader-experience evaluation

---

## How to Route

| Review Type | File | Trigger Keywords |
|-------------|------|------------------|
| `structure` | `subsections/structural-review.md` | organization, flow, structure, narrative arc, transitions, paragraph |
| `content` | `subsections/content-review.md` | facts, accuracy, claims, evidence, methodology, completeness |
| `style` | `subsections/style-review.md` | grammar, spelling, tone, voice, wording, clarity, punctuation |
| `general` | `subsections/general-review.md` | overall, impression, cohesion, purpose, reader experience |

---

## Output Format

For each review, produce a structured report with:

1. **Summary** — One paragraph on the document's overall state in this dimension
2. **Issues Found** — Numbered list of specific problems with line references
3. **Patterns to Address** — Recurring issues worth systemic fixes
4. **Strengths** — What works well (don't just flag problems)
5. **Recommendations** — Prioritized list of what to fix first

ALWAYS point to specific textual content, NOT LINE NUMBERS, so when the file is edited the maked issues can still be detected.
