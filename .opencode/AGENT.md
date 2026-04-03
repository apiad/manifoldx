# AGENT.md — Core Agent Constitution

This file is injected into every agent. It defines universal mandates,
conventions, and registries that apply everywhere.

---

## Universal Mandates

Every agent MUST follow these:

1. **Evidence-based** — Every claim must cite specific sources (files, docs, web)
2. **Explicit before implicit** — State assumptions, plans, and intentions clearly
3. **Read-only default** — Do not modify files unless explicitly authorized
4. **No subagents by default** — Only invoke subagents if explicitly allowed in your agent definition
5. **No file writes without protocol** — Document creation follows your agent's protocol
6. **Context before action** — Always understand before jumping to solutions

---

## Directory Conventions

| Path | Purpose | Created by |
|------|---------|------------|
| `.knowledge/notes/*` | Analysis outputs (research, audits, investigations) | analyze agent |
| `.knowledge/plans/*` | Action plans | design agent |
| `.knowledge/design/*` | Design documents and architecture | design agent |
| `.knowledge/reports/*` | Analysis reports | analyze agent |
| `.playground/*` | Subagent scratch space, experiments | subagents, analyze |
| `docs/` | Project documentation | create agent |

---

## Tool Assumptions

### Always Available
- `read`, `grep`, `glob` — File inspection
- `edit`, `write` — File modification (requires permission)
- `question` — Request user clarification

### External Dependencies
| Tool | Used by | Required? |
|------|---------|-----------|
| `git` | create | Yes |
| `gh` | analyze | No |
| `make` | create | Yes |
| `uv`/`npm`/`cargo` | project-specific | No |

---

## Sandbox System

All shell commands run in Docker containers for isolation. The sandbox plugin
automatically routes commands through `.opencode/sandbox/sandbox.sh`.

### Mount Matrix

| Mode | Project | .playground | .knowledge |
|------|---------|------------|------------|
| analyze | :ro | :rw | :rw |
| design | :ro | :ro | :rw |
| create | :rw | :rw | :rw |

### Sandboxed Commands

Commands that need isolation (language runtimes, installers, build tools):
- `python`, `pip`, `uv`
- `node`, `npm`, `yarn`, `pnpm`, `bun`
- `cargo`, `rustc`
- `go`
- `make`, `cmake`

### Host Commands

These run directly on the host (fast, no isolation):
- `ls`, `grep`, `find`, `cat`, `head`, `tail`
- `git status`, `git log`, `git diff`, `git show`
- `pwd`, `which`, `cd`

**NOTE:** All shell commands must be run with **relative paths** rooted at the project base directory. This is because shell commands are run inside a Docker container, in `/project` workdir, so all relative paths inside the project are kept, but absolute paths, and any path outside the project are disabled.

---

## Primary Mode Registry

### Analyze Mode
- **Purpose**: Understand, investigate, research
- **Permissions**: Read-only on project files; write to `.knowledge/notes/`, `.knowledge/reports/`, `.playground/`
- **Sandbox**: Project :ro, .playground :rw, .knowledge :rw
- **Subagents**: scout, investigator, critic

### Design Mode
- **Purpose**: Design architecture, create plans, make strategic decisions
- **Permissions**: Read-only on project files; write to `.knowledge/plans/`, `.knowledge/design/`
- **Sandbox**: Project :ro, .playground :ro, .knowledge :rw
- **Subagents**: investigator

### Create Mode
- **Purpose**: Implement, create, make changes
- **Permissions**: Full write access to project files
- **Sandbox**: Project :rw, .playground :rw, .knowledge :rw
- **Subagents**: tester, drafter, general

---

## Mode Switching

Modes are manual entry points via commands:

| Command | Switches to |
|---------|-------------|
| `/analyze` | Analyze mode |
| `/design` | Design mode |
| `/create` | Create mode |

Users can also trigger mode detection from intent:
- Questions, research → **Analyze**
- Strategy discussions → **Design**
- Implementation requests → **Create**

---

## Subagent Registry

| Subagent | Purpose | Used By | Writes To |
|----------|---------|---------|-----------|
| `scout` | Web research | analyze | Returns to parent only |
| `investigator` | Codebase analysis | analyze, design | Returns to parent only |
| `critic` | Prose review | analyze | Returns to parent only |
| `tester` | Hypothesis validation | create | `.playground/tests/` |
| `drafter` | Content drafting | create | `.playground/drafts/` |

### Subagent Rules

1. **Never write to project files** — Only parent agents commit changes
2. **Never write to `.knowledge/`** — Parent owns knowledge architecture
3. **Can write to `.playground/`** — Scratch space, gitignored
4. **Must return structured output** — Parent synthesizes results
5. **60 second timeout** — Fast feedback
6. **No nesting** — Subagents cannot spawn subagents

---

## Command Registry

### Mode Entry Commands

| Command | Mode | Description |
|---------|------|-------------|
| `/analyze` | analyze | Switch to Analyze mode |
| `/design` | design | Switch to Design mode |
| `/create` | create | Switch to Create mode |

### Analyze Commands

| Command | Description |
|---------|-------------|
| `/research [topic]` | Deep research with parallel scouts |
| `/audit [scope]` | Comprehensive codebase audit |
| `/investigate [problem]` | Root cause analysis |
| `/onboard` | Project orientation |

### Design Commands

| Command | Description |
|---------|-------------|
| `/plan [description]` | Create structured plan |
| `/scaffold [template]` | Generate project structure |

### Create Commands

| Command | Description |
|---------|-------------|
| `/build [feature]` | TCR implementation |
| `/fix [bug]` | Bug fix with regression test |
| `/draft [content]` | Content creation |
| `/commit [message]` | Commit with validation |

### Infrastructure Commands

| Command | Description |
|---------|-------------|
| `/sandbox-setup` | Detect dependencies and build Docker image |

### Freestyle vs Commands

- **Freestyle** (natural language) — Agent responds conversationally in detected mode
- **Commands** (`/research`, `/build`, etc.) — Structured workflows with rich prompts

Both work in any mode, but commands add discipline and constraints.

---

## YAML Frontmatter Standard

All `.knowledge/` files should include:

```yaml
---
id: kebab-case-identifier
created: YYYY-MM-DD
modified: YYYY-MM-DD
type: research | audit | investigation | plan | design | log
status: active | stale | archived
# Optional:
issue: 42
plan: plan-id
sources: [...]
tags: [...]
---
```

---

## Intelligent Decisions (No Hooks)

This framework uses **intelligent decisions** instead of deterministic enforcement:

- **No pre-commit hooks** — Agent decides if journaling is valuable
- **No forced TCR** — Agent explains discipline, user can freestyle
- **No mandatory planning** — Agent suggests, user decides

Agent explains its reasoning. User can override at any time.

---

*Framework Version: 3.0*
