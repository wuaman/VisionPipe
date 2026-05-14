---
name: vp-spec-review
description: Iteratively review and confirm the DEV_SPEC.md of VisionPipe-py, phase by phase. Tracks progress across sessions so you can pick up where you left off. Use this skill whenever the user wants to discuss, review, confirm, or modify the project spec — including phrases like "review the spec", "let's go over DEV_SPEC", "confirm phase X", "继续讨论规范", "spec review", or any mention of reviewing architecture decisions in VisionPipe-py. Also trigger when the user says "vp-spec-review" or asks to resume a previous spec discussion.
---

# VisionPipe-py Spec Review

You are guiding a Python developer through the design decisions in a C++/Python hybrid video AI framework (VisionPipe-py). The spec is organized into phases (Phase 0-5), each containing multiple design decisions that need explicit user confirmation before being finalized.

## Why this process matters

The DEV_SPEC.md is the single source of truth for implementation. Confirming decisions phase-by-phase prevents contradictions from accumulating — a choice in Phase 2 (e.g., "Tensor is move-only") constrains what's possible in Phase 3 (e.g., Python bindings can't copy Tensors). The forward-consistency check catches these conflicts early.

## Core workflow

```
┌─────────────────────────────────────────────────────┐
│  1. Read progress file → find next unconfirmed phase │
│  2. Present decisions → explain C++ in Python terms  │
│  3. Discuss until user confirms all decision points  │
│  4. Forward-check against ALL confirmed phases       │
│  5. Write confirmed decisions to DEV_SPEC.md         │
│  6. Update progress file                            │
└─────────────────────────────────────────────────────┘
```

## Step 1: Check progress

Read `/home/aman/workspace/llm_space/VisionPipe-py/.claude/spec_review_progress.md`.

If the file doesn't exist, create it with this structure:

```markdown
# Spec Review Progress

## Status
Last reviewed: (none)
Next phase: Phase 0

## Confirmed Phases

(none yet)

## Cross-Phase Dependencies

(none yet)
```

Parse the file to determine which phase to start from. Tell the user where you're picking up.

## Step 2: Present the phase

Read the relevant section of `DEV_SPEC.md` at the project root. Extract the key design decisions — these are the choices that constrain future phases or that have multiple valid alternatives.

For each decision point:
1. State the decision clearly
2. Explain any C++ concepts using Python analogies (the user thinks in Python)
3. Explain WHY this decision matters (what it enables or constrains downstream)
4. If there are alternatives, briefly mention them with tradeoffs

### Explaining C++ concepts

The user is a Python developer learning C++ through this project. Focus on:
1. **What the syntax means** — explain the C++ construct itself clearly
2. **Why it's designed this way** — what problem does it solve, what would go wrong without it
3. **Python analogy only when it genuinely helps** — don't force analogies. Sometimes C++ concepts have no clean Python equivalent and it's better to explain them on their own terms

Keep explanations concise. The user is technical and learns fast — they want to understand the reasoning, not be hand-held through every keyword.

## Step 3: Confirm decisions

Use AskUserQuestion or direct conversation to get explicit confirmation on each decision point. The user may:
- Confirm as-is
- Request a modification (update the decision)
- Ask for more explanation
- Defer a decision (mark it as "deferred" in progress)

Once all decision points for the phase are confirmed (or explicitly deferred), move to Step 4.

## Step 4: Forward-consistency check

This is the critical quality gate. Compare the newly confirmed decisions against ALL previously confirmed phases. Check for:

1. **Interface mismatches**: Does a type defined in Phase 0 get used differently in Phase 2?
2. **Naming inconsistencies**: Same concept called different things in different phases?
3. **Architectural conflicts**: Does a Phase 1 decision (e.g., "nodes are single-threaded") contradict a Phase 2 decision (e.g., "InferNode has parallel workers")?
4. **Dependency violations**: Does the new phase assume something that an earlier phase explicitly ruled out?
5. **Data flow breaks**: Does the Frame structure from Phase 0 have all the fields that Phase 2 nodes expect to read/write?

Report findings to the user. If conflicts exist:
- Explain the conflict clearly
- Propose resolution options
- Get user confirmation on the resolution
- Note whether the resolution requires updating a previously confirmed phase

If a previously confirmed phase needs updating, flag it clearly and update both the phase section in DEV_SPEC.md and the progress file.

## Step 4.5: Code-vs-spec verification

After the forward-consistency check, verify that the **existing implementation** matches the confirmed decisions. Read the relevant source files for the phase being confirmed and check:

1. **Does the code match the spec?** If the spec says "Tensor is move-only RAII" but the code uses raw pointers without cleanup, that's a mismatch.
2. **Are there deviations?** Maybe the code does something reasonable but different from what the spec says (e.g., spec says `std::any` but code uses `map<string, any>`). These need to be reconciled — either update the spec to match the code, or flag the code as needing rework.
3. **Is the task actually complete?** The tracking table (section 6.2) may mark a task as `[x]` but the implementation doesn't match the confirmed decisions.

When mismatches are found:
- Report them clearly to the user with file paths and line numbers
- Ask: "Should we update the spec to match the code, or mark this task as incomplete and flag it for rework?"
- If the user decides the code needs rework: update the tracking table in DEV_SPEC.md section 6.2 to change `[x]` back to `[ ]` for that task, and add a note explaining what needs to change
- If the user decides the spec should match the code: update the spec text accordingly

This step ensures the spec stays grounded in reality — it's not just a theoretical document but reflects what's actually built.

## Step 5: Write to DEV_SPEC.md

After consistency check and code verification pass, update the relevant section in DEV_SPEC.md. Only modify the specific phase section being confirmed. Preserve all other content exactly as-is.

If the discussion resulted in changes to the spec (not just confirmation of existing text), make the edits. If the existing text already matches the confirmed decisions, no edit is needed — just note that the phase is confirmed as-is.

If code-vs-spec mismatches were found and the user chose to mark tasks as incomplete, also update section 6.2 (tracking table) accordingly.

## Step 6: Update progress file

Update `.claude/spec_review_progress.md` with:
- The phase just confirmed
- A brief summary of key decisions (2-3 bullet points per phase)
- Any cross-phase dependencies discovered
- What the next phase to review is

## Handling phase additions and deletions

The user may want to:
- **Insert a new phase** between existing ones (e.g., "I want a Phase 1.5 for X")
- **Delete/merge phases** (e.g., "Phase 4 and 5 should be one phase")
- **Reorder phases**

When this happens:
1. Discuss the change with the user to understand the motivation
2. Propose the new phase structure
3. After user confirms the structural change, re-run the forward-consistency check against ALL previously confirmed phases — the restructuring may invalidate earlier decisions
4. Update both DEV_SPEC.md (phase structure) and the progress file
5. Mark any phases that need re-confirmation due to the structural change

## Session boundaries

This skill is designed to work across multiple sessions. Each session:
- Starts by reading the progress file
- May confirm one or more phases
- Ends by updating the progress file

If the user wants to stop mid-phase, save partial progress by noting which decision points within the phase have been confirmed so far.

## Tone and pacing

- Don't dump all decisions at once. Present 2-4 at a time, get confirmation, then continue.
- If the user seems engaged and wants to go faster, batch more together.
- If the user asks "what does X mean?", explain thoroughly with examples before asking for confirmation.
- Use Chinese when the user writes in Chinese (the spec is bilingual, the user communicates in Chinese).
