# Project conventions

## Prose formatting (thesis / documentation)

- All prose files (`.tex`, `.md` — thesis chapters, experiment write-ups, notes) use **one sentence per line** (semantic line breaks). Never hard-wrap sentences across lines and never put multiple sentences on one line.
- This applies both when writing new prose and when editing existing prose: if a paragraph you touch is not in this format, reflow it.
- Protected content is exempt: LaTeX environments (`table`, `tabular`, `equation`, `figure`), Markdown tables, headings, blockquote captions, and code blocks keep their own layout.
- For bulk conversion use `experiments/phic_psi_poc/thesis/sentence_per_line.py` (`python3 sentence_per_line.py <file>`): joins hard-wrapped paragraphs, splits after sentence-ending punctuation (protecting abbreviations like e.g., i.e., Fig., vs., et al.), treats `\item` and `\paragraph` as new-line anchors, and leaves protected environments untouched.
