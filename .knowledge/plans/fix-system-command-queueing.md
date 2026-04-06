---
id: fix-system-command-queueing
created: 2026-04-06
modified: 2026-04-06
type: plan
status: completed
expires: 2026-04-13
phases:
  - name: Phase 1 - Pass engine to ComponentView
    done: true
    goal: Store engine reference in ComponentView (transparent to user)
  - name: Phase 2 - Pass commands to ComponentAccessor
    done: true
    goal: Pass command buffer from ComponentView to ComponentAccessor
  - name: Phase 3 - Queue commands in _FieldView
    done: true
    goal: Modify _FieldView to queue UPDATE_COMPONENT instead of direct writes
  - name: Phase 4 - Queue commands in ComponentAccessor
    done: true
    goal: Modify ComponentAccessor bulk operations to queue commands
  - name: Phase 5 - Verify and test
    done: true
    goal: All examples work and tests pass
---

# Plan: Fix System Command Queueing

## Context

Systems modify component data directly via `_FieldView` and `ComponentAccessor` instead of queuing UPDATE_COMPONENT commands. This causes the render pipeline to read stale data since commands execute BEFORE the render pipeline reads.

The current broken flow:
1. Systems run → modify data directly (direct write)
2. Command buffer execute → nothing to do  
3. Render pipeline runs → reads same data as before system ran → no updates visible!

Correct flow should be:
1. Systems run → queue UPDATE_COMPONENT commands
2. Command buffer execute → apply updates to components
3. Render pipeline runs → reads updated data → changes visible!

## Phases

### Phase 1: Pass Engine to ComponentView
**Goal:** Store engine reference in ComponentView (transparent to user, via existing `@engine.system` decorator)

**Deliverable:** Modified `ComponentView.__init__()` that accepts engine

**Done when:**
- [ ] `ComponentView` stores engine reference
- [ ] Engine passes itself when creating ComponentView
- [ ] User doesn't see engine in system function signature

**Depends on:** None

### Phase 2: Pass Commands to ComponentAccessor
**Goal:** Pass command buffer from ComponentView to ComponentAccessor

**Deliverable:** Modified `ComponentView.__getitem__()` that passes commands to ComponentAccessor

**Done when:**
- [ ] `ComponentAccessor` receives command buffer reference
- [ ] Accessor can queue commands without user knowing

**Depends on:** Phase 1

### Phase 3: Queue Commands in _FieldView
**Goal:** `_FieldView._set_data()` queues UPDATE_COMPONENT instead of direct write

**Deliverable:** Modified `_FieldView` that stores command buffer and queues updates

**Done when:**
- [ ] `_FieldView` stores reference to command buffer
- [ ] `_set_data()` queues UPDATE_COMPONENT command instead of writing
- [ ] `__iadd__`, `__isub__`, `__imul__` use updated `_set_data()`
- [ ] Validation warnings still work

**Depends on:** Phase 2

### Phase 4: Queue Commands in ComponentAccessor
**Goal:** `ComponentAccessor` also queues commands for bulk updates

**Deliverable:** Modified `ComponentAccessor` that queues UPDATE_COMPONENT

**Done when:**
- [ ] `__iadd__`, `__imul__`, `__itruediv__` queue UPDATE_COMPONENT commands
- [ ] Validation warnings still work

**Depends on:** Phase 3

### Phase 5: Verify and Test
**Goal:** Confirm command execution order and all tests pass

**Deliverable:** Working examples with proper command queueing

**Done when:**
- [ ] `engine._draw_frame()` order confirmed: systems → commands → render
- [ ] `uv run python examples/cube.py` shows rotating cube moving up
- [ ] `uv run python examples/cubes.py` shows cubes spawning and moving
- [ ] All 101 tests pass

**Depends on:** Phase 4

## Success Criteria
- Systems never write directly to component data
- All component modifications go through command buffer
- Render pipeline sees updated data from systems
- Examples show objects moving/rotating correctly

## Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Breaking existing code with new signature | Low | Only add field, don't change existing |
| Command queue causing lag with many entities | Medium | Batch updates per component |
| Validation warnings breaking with commands | Low | Test after each phase |

## Related
- Investigation: Investigator task (ses_29d111887ffefdulTrVEPqVFO1)
- Root cause: _FieldView._set_data() writes directly
- Symptom: Transform positions not updating in render