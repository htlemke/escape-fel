# Did you know?

Short tips on escape features, newest first. Each one gives the main usage and
what you gain from it, with a link to the full documentation. The collection
so far covers mostly features added since May 2026. Tips marked
*experimental* are about new modules that are ready for testing but not yet
validated widely.

```{toctree}
:glob:
:reversed:
:maxdepth: 1

20*
```

## Adding a tip

One Markdown file per tip in this folder, named
`<YYYY-MM-DD>-<short-slug>.md`. The date is when the feature landed, so the
list above sorts itself. The format:

````markdown
---
added: 2026-09-12
tags: [plotting, analysis]
details: user_guide/plot_tools
---

# A one-line title that states the gain

One to three sentences: what it does for you and how you use it.

```python
the_main_call()
```

More: {doc}`../user_guide/plot_tools`
````

- **Keep it short**: around 60 words of prose plus one code block of one to
  three lines.
- **The title states the benefit**, not the implementation ("Reloading a run
  is about ten times faster", not "parse-result cache").
- **`details`** is the Sphinx document name (relative to `docs/`, without
  the extension) of the full page. For a feature that needs more than a tip,
  write that page and link it.
- **`tags`** are free-form lowercase words (`plotting`, `stream`, `storage`,
  ...) for filtering; add `experimental` for modules not yet validated.
- The front matter is plain YAML, so a tool such as a "tip of the day" can
  read the files without Sphinx.
