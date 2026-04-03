---
description: Deep research on a topic using parallel scouts
agent: analyze
---

# /research

Execute comprehensive research using parallel scouts.

## Workflow

### Phase 1: Research plan

Define and present the user a comprehensive research plan with 2-6 research questions / objectives. Iterate with the user until the plan is accepted.

If necessary, perform a minimal set of targeted `websearch` or `read` to gather context for the research.

Use `todowrite` to keep track of open questions.

Create `.knowledge/notes/research-{slug}/` to store temporary notes with boilerplate markdown notes for each research question. Write down in each note the minimal context and relevant subquestions to address.

### Phase 2: Scout Deployment

**Perform the following iteration several times until satisfied:**

Launch `scout` subagents in parallel for each research question. Point each scout to the corresponding research note, where they will write their output.

**Evaluate** for every scout that finishes if further investigation is necessary on any research questions, and repeat until satisfied.

If necessary, use `todowrite` to add follow up relevant questions, and iterate.

### Phase 3: Documentation

Read all notes files in `.knowledge/notes/{slug}/*.md`.

Write structured research to `.knowledge/notes/report-{slug}.md`:

```markdown
---
id: {kebab-case-topic}
created: {date}
type: research
status: active
sources: [list all scout sources]
---

# Research: [Topic]

## Executive Summary
[2-3 paragraphs maximum]

## Key Findings
1. **[Finding 1]**
   - Evidence: [from which scout]
   - Implication: [what this means]
   - Link to relevant section in notes markdown.
   - Link to external sources.

2. **[Finding 2]**
   ...

## Recommendations
- [Specific, actionable recommendation]
- [Specific, actionable recommendation]

## Further reading
[Suggest follow up reading on scout-reported sources for adjacent topics]

## Follow-up questions

[Suggest follow-up research questions that weren't explored.]
```

## Key Constraints
- 60s timeout per scout
- Must synthesize, not just aggregate
- Include conflicting viewpoints if found
- Be thorough and detailed, leave no gaps.
