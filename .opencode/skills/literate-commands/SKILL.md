---
name: literate-commands
description: Create guided, multi-step workflows that walk users through complex tasks with variable collection, conditional logic, and automated script execution. Extends regular opencode commands.
---

# Literate Commands Skill

Use this skill when you need to create **guided, multi-step workflows** that walk users through complex tasks with variable collection, conditional logic, and automated script execution.

## When to Use Literate Commands

Use literate commands when you want to:

- **Guide users through multi-step processes** (setup wizards, onboarding flows)
- **Collect structured data** from users before executing commands
- **Create reusable command templates** with customizable parameters
- **Build conditional workflows** that branch based on user input
- **Automate script execution** with variable interpolation

**Don't use them for:** Simple one-off commands that just run a single script or tool.

## Creating a Literate Command

1. Create a markdown file in `.opencode/commands/`
2. Add `literate: true` in the frontmatter
3. Define steps separated by `---`

```markdown
---
description: My guided workflow
literate: true
---

First step content...

---

Second step content...
```

## Step Configuration

Each step can have a YAML config block to control behavior:

```yaml {config}
step: unique-step-name
parse:
  variable: type
next:
  "condition": target-step
stop: true
```

### Config Options

| Option | Type | Description |
|--------|------|-------------|
| `step` | string | Unique name for routing (use `kebab-case`) |
| `parse` | object | Variables to extract from user response |
| `next` | string | object | Routing rules after this step |
| `stop` | boolean | End command after this step |
| `exec` | object | Script execution settings |

### Step Examples

**Collect data:**

```yaml
parse:
  username: string
  age: number
  newsletter: bool
```

**Route conditionally:**

```yaml
next:
  "role === 'admin'": admin-panel
  "confirmed": proceed
  _: fallback
```

**End command:**

```yaml
stop: true
```

## Variable Collection (`parse`)

Ask the model to respond with JSON containing specific variables:

```yaml
parse:
  name: string
  count: number
  active: bool
```

The model will be prompted to respond with:

```json
{"name": "...", "count": 0, "active": true}
```

### Type Coercion

| Type | Description |
|------|-------------|
| `string` | Convert to text |
| `number` | Convert to numeric |
| `bool` | Convert to true/false |

## Variable Substitution

Use `$variable` in your prompts to inject collected data:

| Syntax | Example | Result |
|--------|---------|--------|
| `$name` | `$name` | Alice |
| `$obj.prop` | `$user.email` | alice@example.com |
| `$arr.0` | `$items.0` | first item |
| `$$` | `$$` | Full metadata JSON |

### Example

```markdown
Hello **$name**! You selected $count items.
```

If `name=Alice` and `count=42`:

```
Hello **Alice**! You selected 42 items.
```

## Routing (`next`)

Control which step executes next based on variables:

### Simple Redirect
```yaml
next: other-step
```

### Conditional Routing

```yaml
next:
  "role === 'admin'": admin-panel
  "role === 'user'": user-panel
  _: default-panel
```

The `_` key is the fallback when no condition matches.

### Condition Syntax

Use JavaScript-style expressions:

- `"confirmed === true"`
- `"count > 10"`
- `"role !== 'guest'"`
- `"name.includes('admin')"`

## Script Execution (`exec`)

Run scripts within steps. The agent sees the result, or you can store variables.

### Basic Execution

```bash {exec}
echo "Hello $name"
```

### Store Output as Variables

```python {exec mode=store}
import json
print(json.dumps({"result": len("$name")}))
```

### Exec Modes

| Mode | Description |
|------|-------------|
| `stdout` | Show output to user (default) |
| `store` | Parse JSON output and store as variables |
| `none` | Execute silently (no output) |

### Interpreters

Specify the interpreter explicitly if its not the default:

```python {exec=uv run python -c}
print("Python!")
```

```bash {exec=sh -c}
echo "Bash!"
```

```javascript {exec=node -c}
console.log("Node!");
```

## Complete Example

    ---
    description: Project setup wizard
    literate: true
    ---

    ```yaml {config}
    step: project-name
    parse:
      name: string
      type: string
    ```

    What would you like to name your project, and what type is it?

    ---

    ```yaml {config}
    step: confirm
    next:
      "confirmed === true": create
      _: cancel
    ```

    Ready to create **$name** ($type)? Type `true` or `false`.

    ---

    ```yaml {config}
    step: create
    parse:
      confirmed: bool
    ```

    Let me set that up for you:

    ```bash {exec mode=store}
    mkdir -p "$name"
    echo '{"created": true, "path": "'$name'"}'
    ```

    ---

    ```yaml {config}
    step: success
    stop: true
    ```

    ✓ Project **$name** created successfully!

    ---

    ```yaml {config}
    step: cancel
    stop: true
    ```

Project creation cancelled. Let me know if you need anything else!

## Tips

1. **Keep steps focused** - One clear action per step
2. **Use meaningful step names** - Makes routing easier to debug
3. **Validate early** - Collect and validate data before expensive operations
4. **Store, don't echo** - Use `mode=store` for script results you want to keep
5. **Test routing** - Use `stop: true` on branches to verify they work

## Testing Your Command

```bash
opencode run --print-logs --log-level DEBUG --command your-command-name
```

The command will run through steps sequentially. Check logs for:
- Step parsing
- Variable substitution
- Routing decisions
- Script execution
