---
description: Root cause analysis for bugs and issues
agent: analyze
subagents: [investigator, tester]
---

# /investigate [problem]

Scientific investigation of [problem] to determine root cause.

## Workflow

### Phase 1: Problem Definition
Clarify:
- What is the observed behavior?
- What is the expected behavior?
- When does it occur? (reproduction steps)
- What is the impact?

### Phase 2: Context Gathering
Use `investigator` to understand:
- Relevant code paths
- Recent changes
- Dependencies involved
- Environment factors

### Phase 3: Hypothesis Formation
Generate potential causes:
1. Hypothesis A: [possible cause]
2. Hypothesis B: [possible cause]
3. Hypothesis C: [possible cause]

### Phase 4: Testing
Use `tester` subagent to validate hypotheses:
- Create minimal reproduction
- Test each hypothesis
- Narrow to root cause

### Phase 5: RCA Report
Write investigation to `.knowledge/insights/investigations/{slug}.md`:

```yaml
---
id: investigation-{slug}
created: {date}
type: investigation
status: resolved
severity: [critical/high/medium/low]
---

# Investigation: [Problem]

## Problem Statement
[Clear description of issue]

## Symptoms
- [Observable symptom]
- [Observable symptom]

## Root Cause
[What actually caused the problem]

## Evidence
- [Specific file/line evidence]
- [Test results]

## Resolution
[How to fix it]

## Prevention
[How to prevent recurrence]
```

## Key Constraints
- Scientific method — hypothesis → test → conclude
- Minimal reproduction — smallest test case
- No fixes yet — ANALYZE mode is understanding only
