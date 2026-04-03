---
description: Create content with iterative drafting
agent: create
subagents: [drafter, critic]
---

# /draft [content brief]

Create content from [content brief] using iterative drafting.

## Workflow

### Phase 1: Structure
Define content outline:
- Sections and their purposes
- Target length per section
- Key points to cover

### Phase 2: Parallel Drafting
Use `drafter` subagents (parallel):
- Drafter 1: Section A
- Drafter 2: Section B
- Drafter 3: Section C

### Phase 3: Review
Use `critic` subagent to review:
- Overall coherence
- Section transitions
- Style consistency
- Clarity and completeness

### Phase 4: Refinement
- Address critic feedback
- Smooth transitions between sections
- Final polish

### Phase 5: Output
Write to appropriate location:
- `drafts/` for work in progress
- `docs/` for documentation
- Project root for README, etc.

## Key Constraints
- Multiple drafters for parallel work
- Critic review before finalizing
- Iterative refinement
- Match project style guide
