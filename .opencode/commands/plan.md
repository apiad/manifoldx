---
description: Create structured, actionable plan with YAML frontmatter
agent: design
subagents: [investigator]
---

# /plan

Create structured implementation plan.

## Workflow

### Phase 1: Constraint Analysis

Use `investigator` to understand:
- Existing code structure
- Relevant patterns to follow
- Technical constraints
- Dependencies involved

### Phase 2: Decomposition

Break [description] into phases:
1. **Phase 1:** [initial step]
2. **Phase 2:** [next step]
3. **Phase 3:** [final step]

Maximum 5 phases for clarity.

### Phase 3: Definition

For each phase, define:
- **Goal:** What must be achieved
- **Deliverable:** Tangible output
- **Done when:** Clear completion criteria
- **Depends on:** Previous phases

### Phase 4: Documentation

Write plan to `.knowledge/plans/{kebab-case-description}.md`:

```yaml
---
id: {kebab-case-description}
created: {date}
modified: {date}
type: plan
status: active
expires: {date+7days}
phases:
  - name: Phase 1 Name
    done: false
    goal: What this phase achieves
  - name: Phase 2 Name
    done: false
    goal: What this phase achieves
---

# Plan: [Description]

## Context
[Why this plan exists, background]

## Phases

### Phase 1: [Name]
**Goal:** [What must be achieved]

**Deliverable:** [Tangible output]

**Done when:**
- [ ] [Criterion 1]
- [ ] [Criterion 2]

### Phase 2: [Name]
...

## Success Criteria
[How we know the plan is complete]

## Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| [what could go wrong] | [high/med/low] | [how to handle] |

## Related
- Issue: #42 (if applicable)
- Research: [link to insight]
```

## Key Constraints

- Maximum 5 phases
- Clear done criteria for each
- Expires in 7 days (warns when stale)
- Must be actionable — no vague phases
