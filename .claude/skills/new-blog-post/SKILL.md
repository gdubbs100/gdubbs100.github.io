---
name: new-blog-post
description: Scaffold a new blog post — creates the post file in _posts and a matching images folder, following this site's existing conventions. Use when the user asks to create, start, or draft a new blog post.
---

# New Blog Post

Scaffolds a new Jekyll post in `_posts/` plus a matching, empty folder in `images/`, following the conventions already established by the posts in this repo.

## Existing conventions (derived from current posts)

- Post filename: `_posts/YYYY-MM-DD-<slug>.md`, e.g. `2026-04-12-battery.md`, `2026-01-24-MPC.md`.
- Front matter is minimal — just one field:
  ```yaml
  ---
  overview: "<one-sentence summary of the post>"
  ---
  ```
  The page title comes from the first `# Heading` in the body (see `titles_from_headings` in `_config.yml`), not from front matter.
- Image folder: `images/<YYYYMMDD><Slug>/` — the post's date with no dashes, immediately followed by the same slug used in the filename (capitalization included), e.g. `20260412Battery`, `20260221Cauchy`, `20260124MPC`, `20260617tictactoe`.
- Images are referenced in-post as `![alt text](/images/<YYYYMMDD><Slug>/<file>.png)` (absolute path from site root — this is the convention used in the most recent post; older posts used relative `../../../images/...` paths, but prefer the absolute form for new posts).

## Steps

1. **Ask the user for the post title** if they haven't already given one.
2. **Ask the user for a one-sentence overview** for the `overview` front matter field. If they don't have one yet, offer to draft a placeholder they can edit later — don't invent post content on their behalf.
3. **Determine today's date** in `YYYY-MM-DD` format.
4. **Derive a slug** from the title, short and simple (matching the style of existing slugs like `MPC`, `cauchy`, `battery`, `tictactoe` — not a long hyphenated phrase). Confirm the slug with the user before creating files if it isn't obvious from the title.
5. **Create the post file** at `_posts/<YYYY-MM-DD>-<slug>.md`:
   ```
   ---
   overview: "<overview text>"
   ---
   # <Title>

   ```
6. **Create the matching image folder** at `images/<YYYYMMDD><slug>/` (no dashes in the date, empty — the user will drop images into it as they write).
7. **Report the two paths created** and remind the user to reference images with `![alt](/images/<YYYYMMDD><slug>/<file>.png)`.

## Notes

- Don't write post content — only scaffold the file and folder; the body is left for the user to fill in.
- Don't add extra front matter fields (no `title`, `date`, or `layout`) — the existing posts don't use them, and Jekyll infers what it needs.
