---
description: Detect project dependencies and build sandbox Docker image
agent: create
literate: true
---

```python {exec}
import pathlib

basename = pathlib.Path(".").resolve().name

print(f"Base dockerfile template: **.opencode/sandbox/dockerfile**")
print(f"This project custom dockerfile: **.opencode/sandbox/{basename}.dockerfile**")
print(f"Project sandbox image name: **{basename}-sandbox**")
```

We are going to build or update the custom dockerfile image so we can run ALL tools and commands necessary for this project inside that image. This includes all general shell commands with side effects (like git, make, etc.) and all project-specific commands to run code, run tests, deploy, etc.

First, look at the base dockerfile template and the project custom dockerfile (if it exists), to understand the current state and determine what is already installed.

Present a list of current dependencies.

---

Now scan the project for language runtimes and build tools:

- Language-specific toolchains like Python, Go, Node, Rust, C/C++, Java, C#, etc.
- General tools like LaTeX, Quarto, etc.
- Project-specific dependencies in makefile.
- System utilities not available by default that are REQUIRED by this project.
- Everything else that requires some system dependencies.

DO NOT simply report everything that may be needed in the future. Focus on dependencies that are demonstrably necessary in the current state of the project.

Make a thorough list identifying everything that is not in the Dockerfile already. Report what must be added, updated, and removed.

---

Based on the previous list, create or regenerate the project custom dockerfile with:

- The base image template from **.opencode/sandbox/dockerfile**.
- Language runtimes that you detected via `COPY --from=` if available, manually if not.
- System or build tools and other toolchain dependencies.
- Any other custom dependency that must be apt-installed or curl | bash installed.

ENSURE you only install what you already determined in the previous message that was necessary. Make no further additions in this stage.

---

Finally run `docker build` to create the project sandbox image.

---

Report what was detected and whether the build succeeded. Ask for further instructions if anything failed.
