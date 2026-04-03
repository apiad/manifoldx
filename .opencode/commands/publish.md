---
description: Version, tag, and deploy release
agent: create
subagents: []
---

# /publish [version]

Release version [version] with changelog and deployment.

## Workflow

### Phase 1: Pre-flight Checks
- Working tree clean?
- On main branch?
- All tests pass (`make test`)?
- Version not already tagged?

### Phase 2: Changelog
Generate changelog from commits since last tag:
```
## [version] - {date}

### Features
- [feat commits]

### Fixes
- [fix commits]

### Other
- [other commits]
```

Append to CHANGELOG.md.

### Phase 3: Version Bump
Update version in:
- `pyproject.toml` or `package.json`
- Version constants if applicable

Commit: `git commit -m "chore(release): version [version]"`

### Phase 4: Tagging
```bash
git tag -a v[version] -m "Release [version]"
```

### Phase 5: Deployment
Run `make deploy` (project-defined):
- Push to registry
- Deploy to staging/prod
- Update documentation

### Phase 6: Finalize
Push commits and tags:
```bash
git push origin main
git push origin v[version]
```

Update `.knowledge/log/{date}.yaml`:
```yaml
entries:
  - timestamp: {iso}
    type: release
    version: [version]
    changelog: [summary]
```

## Key Constraints
- Tests must pass
- Working tree must be clean
- User confirmation at each major step
- Irreversible external actions clearly marked
