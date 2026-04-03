---
description: Commit changes with validation
agent: create
subagents: []
---

# /commit [message]

Commit current changes with validation.

## Workflow

### Phase 1: Validation
Run `make test`:
- If pass → proceed
- If fail → report failures, ask user

### Phase 2: Staging
Show user what will be committed:
```
Files to commit:
- modified: src/auth.py
- new: tests/test_auth.py
```
Ask for confirmation.

### Phase 3: Commit
Use conventional commit format:
```bash
git add -A
git commit -m "[type]([scope]): [message]"
```

Types: feat, fix, docs, refactor, chore, test

### Phase 4: Log Update
Update `.knowledge/log/{date}.yaml`:
```yaml
entries:
  - timestamp: {iso}
    type: commit
    message: "[message]"
    files: [list]
    issue: #42 (if linked)
```

### Phase 5: Issue Check
Check if commit message references issue:
- "fixes #42" → suggest closing issue
- Ask user if issue should be closed

## Key Constraints
- Tests must pass (unless user overrides)
- Conventional commit format
- Always update log
- User confirmation for significant changes
