---
description: Comprehensive codebase audit for technical debt and architecture
agent: analyze
subagents: [investigator]
---

# /audit [scope]

Audit [scope] of the codebase for architecture, tech debt, and standards.

## Workflow

### Phase 1: Discovery
Use `investigator` subagent to map:
1. Directory structure and organization
2. Key files and their responsibilities
3. Dependencies and relationships
4. Entry points and APIs

### Phase 2: Analysis
Investigate specific areas:
1. **Architecture Patterns**
   - Consistency of patterns
   - Architectural drift
   - Modularity and coupling

2. **Code Quality**
   - Complexity hotspots
   - Duplication
   - Test coverage

3. **Standards Compliance**
   - Naming conventions
   - Documentation
   - Project structure

4. **Technical Debt**
   - TODOs and FIXMEs
   - Deprecated patterns
   - Known issues

### Phase 3: Report
Write structured audit to `.knowledge/notes/audits/{scope}-{date}.md`:

```markdown
---
id: audit-{scope}-{date}
created: {date}
type: audit
scope: [what was audited]
status: active
---

# Audit: [Scope]

## Executive Summary
[Overall health assessment]

## Architecture Overview
[Structure, patterns, entry points]

## Findings

### Strengths
- [What's working well]

### Concerns
| Area | Severity | Description | Recommendation |
|------|----------|-------------|----------------|
| [area] | [high/med/low] | [what] | [fix] |

### Technical Debt
- [Specific debt items with locations]

## Recommendations
1. [Priority recommendation]
2. [Secondary recommendation]
```

## Key Constraints
- Read-only — never modify files
- Evidence-based — cite specific files/lines
- Actionable — every finding has recommendation
