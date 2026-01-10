---
name: skill-maker
description: Generate production-ready Claude Code skills following official documentation standards. Use when creating new skills, scaffolding skill structures, or standardizing skill documentation. Handles validation, clarification, and file generation for skills.
---

# Skill Maker

Generate high-quality, standardized Claude Code skills that follow official documentation best practices. This skill ensures all generated skills are production-ready, properly structured, and fully compliant with Claude Code standards.

## Instructions

### Phase 1: Information Gathering

When a user requests a new skill, gather the following required information through targeted questions:

#### 1. Skill Identity
- **Skill name**: Validate it uses only lowercase letters, numbers, and hyphens (max 64 characters)
- **Purpose**: What problem does this skill solve?
- **Target use cases**: When should Claude invoke this skill?

#### 2. Skill Description
Craft a description (max 1024 characters) that includes:
- **What it does**: Specific capabilities and actions
- **When to use it**: Trigger keywords, file types, technologies, or contexts
- **Key technologies**: Relevant frameworks, tools, or formats

Example: "Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction."

#### 3. Functional Requirements
- **Instructions**: Step-by-step guidance for Claude
- **Examples**: Concrete usage scenarios
- **Dependencies**: Required packages, tools, or external services
- **Tool restrictions**: Should this skill have limited tool access? (read-only, file operations only, etc.)

#### 4. Structural Requirements
- **Complexity level**: Simple (single SKILL.md) or complex (multi-file with references/scripts)?
- **Supporting files needed**:
  - Reference documentation (REFERENCE.md)
  - Extended examples (EXAMPLES.md)
  - Helper scripts (scripts/ directory)
  - Templates (templates/ directory)

### Phase 2: Validation and Clarification

Before generating the skill, validate all requirements:

1. **Name validation**:
   - Check format: lowercase, numbers, hyphens only
   - Verify length: max 64 characters
   - Ensure uniqueness and clarity

2. **Description validation**:
   - Confirm it includes both "what" and "when"
   - Verify trigger keywords are present
   - Check length: max 1024 characters
   - Ensure discoverability and specificity

3. **Completeness check**:
   - Are instructions clear and actionable?
   - Are examples concrete and realistic?
   - Are dependencies specified?
   - Is the structure appropriate for complexity?

**If any information is missing, unclear, or ambiguous**: Ask targeted clarification questions using the AskUserQuestion tool before proceeding.

**Do not proceed to generation until all gaps are resolved.**

### Phase 3: Skill Generation

Once all information is validated and complete:

#### Step 1: Create Directory Structure

```bash
mkdir -p .claude/skills/[skill-name]
```

For complex skills with supporting files:
```bash
mkdir -p .claude/skills/[skill-name]/scripts
mkdir -p .claude/skills/[skill-name]/templates
```

#### Step 2: Generate SKILL.md

Create the main SKILL.md file with proper YAML frontmatter:

```yaml
---
name: skill-name
description: Specific description including what and when (max 1024 chars)
allowed-tools: Optional,Tool,Names
---

# Skill Title

## Instructions

[Clear, numbered steps with rationale]

1. [First step with context]
2. [Second step with explanation]
3. [Third step with expected outcome]

## Examples

### Example 1: [Scenario Name]

[Concrete example with code/commands]

### Example 2: [Another Scenario]

[Additional example showing different use case]

## Requirements (if applicable)

- Dependencies or prerequisites
- External tools or packages
- Configuration needed

## Advanced Usage (if applicable)

- Edge cases
- Performance considerations
- Troubleshooting tips
```

#### Step 3: Generate Supporting Files (if needed)

**For REFERENCE.md**:
- Detailed API documentation
- Advanced configuration options
- Complete parameter references

**For EXAMPLES.md**:
- Extended real-world examples
- Complex workflows
- Integration scenarios

**For scripts/ directory**:
- Helper scripts in appropriate language
- Include shebang and documentation
- Add error handling

**For templates/ directory**:
- Template files with placeholders
- Configuration templates
- Boilerplate code

#### Step 4: Validation

Verify the generated skill meets standards:

1. **YAML frontmatter**: Valid syntax, correct fields, proper formatting
2. **File paths**: Use forward slashes, correct relative paths
3. **Markdown formatting**: Proper headers, code blocks, lists
4. **Completeness**: All sections present and populated
5. **Clarity**: Instructions are actionable and examples are concrete

### Phase 4: Testing Recommendations

Provide the user with testing guidance:

1. **Restart Claude Code** to load the new skill
2. **Test invocation** by asking questions that match the skill description
3. **Verify behavior** by checking if Claude uses the skill appropriately
4. **Refine description** if Claude doesn't invoke the skill when expected

## Quality Standards

### YAML Frontmatter Requirements
- Opening `---` on line 1
- Closing `---` before markdown content
- Valid YAML syntax (spaces only, no tabs)
- No trailing whitespace after closing `---`

### Naming Conventions
- **Skill directory**: lowercase-with-hyphens
- **Skill name**: lowercase-with-hyphens (max 64 chars)
- **File names**: SKILL.md (uppercase), supporting files lowercase

### Description Best Practices
- **Be specific**: Mention exact technologies, file types, keywords
- **Include triggers**: Words/phrases that should invoke the skill
- **Explain context**: Both what it does and when to use it
- **Avoid generic terms**: Replace "helps with" with specific actions

### Tool Restriction Patterns

**Read-only skills**:
```yaml
allowed-tools: Read, Grep, Glob
```

**File operations only**:
```yaml
allowed-tools: Read, Write, Edit, Glob
```

**Data analysis**:
```yaml
allowed-tools: Read, Grep, Glob, Bash
```

## Common Clarification Questions

Use these questions when information is missing:

### For Naming
- "What would you like to name this skill? (use lowercase letters, numbers, and hyphens only)"
- "This name is [too long/invalid format]. Would you like to use '[suggested-name]' instead?"

### For Description
- "What specific keywords or file types should trigger this skill?"
- "Can you describe when Claude should use this skill versus handling it normally?"
- "What technologies, frameworks, or tools does this skill work with?"

### For Functionality
- "What are the step-by-step instructions Claude should follow?"
- "Can you provide a concrete example of using this skill?"
- "Are there any external dependencies or tools required?"

### For Structure
- "Does this skill need supporting documentation files, or can everything fit in SKILL.md?"
- "Should this skill have restricted tool access for security or focus?"
- "Are helper scripts or templates needed for this skill?"

## Examples of Generated Skills

### Simple Skill Example

**User request**: "Create a skill for generating commit messages"

**Generated structure**:
```
.claude/skills/commit-message-generator/
└── SKILL.md
```

**SKILL.md content**:
```yaml
---
name: commit-message-generator
description: Generate clear, conventional commit messages from git diffs. Use when writing commits, reviewing staged changes, or working with git version control.
---

# Commit Message Generator

## Instructions

1. Run `git diff --staged` to review staged changes
2. Analyze changes for scope, type, and impact
3. Generate commit message following conventional commits format:
   - Type: feat, fix, docs, refactor, test, chore
   - Scope: affected component or module
   - Subject: concise summary (max 50 chars)
   - Body: detailed explanation of what and why

## Examples

### Example 1: Feature Addition

**Staged changes**: New authentication middleware

**Generated message**:
```
feat(auth): add JWT authentication middleware

Implement middleware for validating JWT tokens on protected routes.
Includes token verification, expiration checking, and user extraction.
```

### Example 2: Bug Fix

**Staged changes**: Fix null pointer in user service

**Generated message**:
```
fix(users): handle null email in getUserByEmail

Add null check before accessing email property to prevent
crashes when user data is incomplete.
```

## Best Practices

- Use present tense ("add" not "added")
- Explain what and why, not how
- Keep subject line under 50 characters
- Separate subject from body with blank line
```

### Complex Multi-File Skill Example

**User request**: "Create a skill for PDF processing with form filling capabilities"

**Generated structure**:
```
.claude/skills/pdf-processor/
├── SKILL.md
├── REFERENCE.md
├── EXAMPLES.md
└── scripts/
    └── fill_form.py
```

**SKILL.md content**:
```yaml
---
name: pdf-processor
description: Extract text and tables from PDFs, fill PDF forms, merge and split documents. Use when working with PDF files, forms, document extraction, or PDF manipulation tasks. Requires pypdf and pdfplumber.
---

# PDF Processor

## Quick Start

Extract text from PDF:
```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    page = pdf.pages[0]
    text = page.extract_text()
    print(text)
```

For detailed form filling instructions, see [EXAMPLES.md](EXAMPLES.md).
For complete API reference, see [REFERENCE.md](REFERENCE.md).

## Instructions

1. Install required dependencies:
   ```bash
   pip install pypdf pdfplumber
   ```

2. For text extraction: Use pdfplumber for complex layouts with tables
3. For form filling: Use pypdf's PdfWriter and update_page_form_field_values
4. For merging: Use PdfMerger to combine multiple PDFs

## Requirements

- Python 3.8+
- pypdf >= 3.0.0
- pdfplumber >= 0.9.0

## Common Tasks

- Extract text: `pdfplumber.open(file).pages[n].extract_text()`
- Fill forms: Use `scripts/fill_form.py` helper script
- Merge PDFs: Create PdfMerger and append files
```

## Validation Checklist

Before delivering a generated skill, verify:

- [ ] YAML frontmatter is valid and complete
- [ ] Skill name follows naming conventions (lowercase, hyphens, max 64 chars)
- [ ] Description includes both what and when (max 1024 chars)
- [ ] Description contains specific trigger keywords
- [ ] Instructions are clear, numbered, and actionable
- [ ] Examples are concrete with real code/commands
- [ ] File paths use forward slashes
- [ ] Supporting files are properly referenced
- [ ] Dependencies are documented
- [ ] Directory structure is appropriate for complexity
- [ ] All files follow markdown best practices

## Error Handling

### Common Issues and Solutions

**Issue**: Skill not being invoked by Claude
**Solution**: Enhance description with more specific keywords and trigger terms

**Issue**: YAML parsing error
**Solution**: Check for tabs (use spaces), verify syntax, ensure proper indentation

**Issue**: Invalid skill name
**Solution**: Convert to lowercase, replace spaces/underscores with hyphens, shorten if needed

**Issue**: Overly complex single file
**Solution**: Split into SKILL.md + REFERENCE.md for progressive disclosure

## Distribution Guidance

After generating a skill, provide distribution instructions:

### For Project Skills (Team Sharing)
```bash
git add .claude/skills/[skill-name]
git commit -m "Add [skill-name] skill for [purpose]"
git push
```

### For Personal Skills
Move to personal skills directory:
```bash
cp -r .claude/skills/[skill-name] ~/.claude/skills/
```

### For Plugin Distribution
Consider creating a plugin if the skill:
- Has multiple supporting files
- Requires external dependencies
- Should be distributed to multiple teams
- Needs version management

## Post-Generation Steps

After generating a skill, guide the user:

1. **Review the generated files** for accuracy and completeness
2. **Test the skill** by restarting Claude Code and asking relevant questions
3. **Refine if needed** based on testing results
4. **Document dependencies** if external tools are required
5. **Share with team** if using project skills directory

## Summary

The skill-maker ensures every generated skill is:
- **Standardized**: Follows Claude Code documentation conventions
- **Complete**: Contains all required information and structure
- **Validated**: Meets naming, formatting, and quality requirements
- **Production-ready**: Can be used immediately without modification
- **Discoverable**: Has effective descriptions with trigger keywords
- **Maintainable**: Properly documented with clear examples

Always prioritize clarity, completeness, and compliance with official Claude Code standards.
