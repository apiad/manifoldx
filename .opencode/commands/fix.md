---
description: Fix bug with regression test
agent: create
subagents: [tester]
---

# /fix [bug description]

Fix [bug description] using regression test discipline.

## Workflow

### Phase 1: Understanding
Use `investigator` (if needed) to:
- Locate bug in codebase
- Understand expected vs actual behavior
- Identify root cause

### Phase 2: Reproduction
Use `tester` subagent to:
- Create minimal reproduction test
- Confirm test fails (demonstrates bug)
- Commit test: `git commit -m "test: reproduce [bug]"`

### Phase 3: Fix
Implement minimal fix:
- Smallest change that fixes the bug
- Run reproduction test → should pass
- Run full test suite → should pass
- Commit: `git commit -m "fix: [bug description]"`

### Phase 4: Verification
- Test passes
- No regressions in full suite
- Edge cases considered

## Key Constraints
- Regression test first — always
- Minimal fix — don't over-engineer
- Full test suite passes
- Document in commit message
