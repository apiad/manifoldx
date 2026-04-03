---
description: Help using the OpenCode Opinionated Framework
agent: analyze
---

# OpenCode Opinionated Framework

USER REQUEST: $ARGUMENTS

**Option 1:** If the command is run without arguments:

Explain to the user what this framework is, and how to use it. Do not perform ANY further analysis or read any files, just reply in the following format:

```
## Framework

[explain this framework in 1 paragraph]

## Main Modes (Agents)

[list each primary agent (mode) and its purpose]

### Subagents

[explain subagents are launched for background tasks automatically, list each one and when it's used]

## Commands

[list each command and its purpose]

## Contextual Analysis

[judging from the recent conversation, suggest what the user might want to do, 2-3 commands or instructions to give primary agents]
```

**Option 2:** If the command is run with arguments:

Perform 2-3 targeted reads in .opencode subfolder to better understand how some agent or command works.

Reply to the user in detail.

**Finally:**

Finally, ask the user if they have a further question about the framework. If so, then repeat option 2 as necessary.
