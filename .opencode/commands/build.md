---
description: Implement feature using TCR discipline
agent: create
subagents: [tester]
---

# /build

Implement using Test-Commit-Revert discipline.

## Workflow

### Phase 1: Planning Check
Check `.knowledge/plans/` for relevant plan:
- If found: Load and follow plan phases strictly. Use `todowrite` to keep track of progress.
- If not found: Instruct user to use `/plan` and refuse to continue.

### Phase 2: TCR Loop

For each implementation step:

**Step 1: RED (Write Test)**
- Write failing test defining the step
- Run `make test` to confirm failure
- Commit the test: `git add -A && git commit -m "test: [step description]"`

**Step 2: GREEN (Implement)**
- Launch `general` subagent with focused instructions to implement:
  - Write minimal code to pass test
  - No refactoring yet
  - Run `make test` to confirm pass
- Give the general subagent concrete instructions on what logic to implement.

**Step 3: VERIFY**
- If pass → Commit: `git commit -m "feat: [step description]"`
- If fail → **One fix attempt**
  - Quick fix (5 min max)
  - Run `make test`
  - If pass → Commit
  - If fail → **REVERT**: `git checkout .`
  - Report failure to user

### Phase 3: Completion
- All plan phases complete
- All tests pass
- Working tree clean
- Update plan status to done

## Key Constraints
- Never commit failing tests
- Maximum one fix attempt before revert
- Parent agent owns all commits
- Tester subagent for complex validations
