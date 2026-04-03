---
description: List and review current tasks from GitHub issues
agent: analyze
subagents: []
---

# /todo

List current tasks and priorities from GitHub issues.

## Workflow

### Phase 1: Fetch Issues
Query GitHub for:
- Open issues assigned to user
- Open issues with specific labels
- Recently updated issues

### Phase 2: Categorize
Group by:
- **Blocking** — Must resolve first
- **Active** — Currently working on
- **Next** — Ready to start
- **Backlog** — Future work

### Phase 3: Present
Show formatted task list:

```
## Current Tasks

### Blocking
- #42 [title] (priority)
  [brief description]

### Active  
- #43 [title]
  [brief description]

### Next
- #44 [title]
  [brief description]
```

## Key Constraints
- Read-only — just report, don't modify issues
- Focus on actionable — prioritize what can be done now
- Link to plans — note if issue has linked plan in `.knowledge/plans/`
