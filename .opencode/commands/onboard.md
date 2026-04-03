---
description: Orient a new developer to the project - architecture, standards, and current state
agent: analyze
subagents: [investigator]
---

# /onboard

Orient a new developer to project architecture, standards, and current state.

## Workflow

### Phase 1: Discovery (Parallel)
Use `investigator` subagent for:

1. **Core Docs**
   - Read `README.md` — project purpose
   - Read `makefile` — available commands
   - Check `.knowledge/plans/` — active work

2. **Structure Mapping**
   - List root files and directories
   - Identify source directories
   - Find configuration files

3. **Technology Stack**
   - Detect languages and frameworks
   - Identify key dependencies
   - Find entry points

### Phase 2: Standards Discovery
- How to run the project
- How to test (`make test`?)
- How to contribute
- Project conventions

### Phase 3: Current State
- Recent activity (git log --oneline -10)
- Active plans/work
- Known issues or blockers

### Phase 4: Onboarding Report
Present professional summary:

```
# Project Onboarding

## Project Purpose
[What this project does]

## Architecture & Layout
```
[directory tree]
```

**Key Technologies:** [list]

## Workflow & Standards
- **Run:** [how to start]
- **Test:** [how to test]
- **Build:** [how to build]

## Current State
- **Active Work:** [plans in progress]
- **Recent Activity:** [last few commits]

## First Steps
1. [specific file to read]
2. [specific command to run]
3. [specific issue to explore]
```

Write to `.knowledge/insights/project-overview.md` for future reference.

## Key Constraints
- Concise — high signal, no fluff
- Practical — focus on getting started
- Current — reflects actual state
