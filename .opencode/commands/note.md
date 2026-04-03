---
description: Structure markdown notes from thoughts or context
agent: analyze
---

# /note

You are running the /note command. This command creates a structured markdown note with a YAML header from the user conversation.

Determine what should be the content of the note (a well-structured markdown body), the note title, slug (kebab-cased filename) and potential tags.

## What to do next

Run the following command with appropriate args.

```bash
uv run .opencode/bin/note.py create [...] << 'EOF'
Note content
EOF
```

**Start here:**

!`uv run .opencode/bin/note.py create -h`

IMPORTANT: Do NOT save the note until the user approves.

**Read the output and follow instructions from there.**
