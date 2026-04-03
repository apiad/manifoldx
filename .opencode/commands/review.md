---
description: Review text or documentation for coherence, style, and grammar.
agent: analyze
literate: true
skill: editor
---

```yaml {config}
parse:
    file: string
```

**Comand arguments:** $ARGUMENTS

First determine the exact path of the document to review from the conversation or ask the user to provide it, using the `question` tool.

Verify the file exists. If in doubt, look around and determine ONE file to review.

Reply "file": "relative/path/to/file.md" when ready.

---

```yaml {config}
parse:
    kind: string
```

First determine what kind of review this is:
 - **structure** review (to analyze the overal narrative arc, how main points connect, etc.),
 - **content** review (to analyze the content factuality, methodology, etc.,) or
 - **style** review (to analyze the grammar, wording, spelling, tone, etc.) or
 - **general** holistic review to be performed as a final resort.

Reviews should be performed in that order, it makes no sense to review low-level stuff when there are high-level issues.

If the current conversation makes it explicit, or there are .review.md files next to $file that let you decide, then just reply what's the next review.

Otherwise, use `question` to ask the user which kind of review to perform.

Reply with "review_kind": "structure|content|style|general"

---

Read the article $file once to get a full view and summarize it briefly.

---

Activate the `editor` skill using the `skill` tool.
Load the corresponding subsection to $review_kind review.
You will review the document following that checklist.
For now just restate the checklist for now.

---

Use `task` to launch parallel `critic` subagents to perform the review, one separate subagent for each main category of issues.

Pass each subagent a concrete list of issues to look for, and a link to the file: $file.

DO NOT PASS the file content, just a link. They will read it again.

---

Write the review report next to $file. Make sure its detailed and points to all concrete issues, not just general ones.

Name it like $file with .review.$review_kind ending instead, markdown file.
