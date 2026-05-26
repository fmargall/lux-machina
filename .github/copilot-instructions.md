# Commit message convention

## Format

Every commit message must follow this exact structure:

    <gitmoji> <type>(<scope>): <short description>
    
    [optional body]

    [optional footer]

## Allowed commit types

After the emoji and a space, always use one of the following commit types with its associated Gitmoji:

 - `feat: <✨>` for new features
 - `fix: <🐛>` for bug fixes
 - `ci: <👷>` for continuous integration
 - `docs: <📝>` for documentation
 - `style: <🎨>` for code style improvements
 - `refactor: <♻️>` for refactoring
 - `perf: <⚡️>` for performance improvements
 - `test: <✅>` for tests
 - `chore: <🔧>` for anything else

Use the most appropriate commit type and its associated Gitmoji taken from this list. Never use another type.

## Style guidelines

- Write commit messages in English
- Keep the description concise and imperative
- Do not end the description with a period
