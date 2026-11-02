# Decisions

Section 9 decisions are in the approved plan. This file logs what came after it, and gets copied to Drive `00_admin/decisions.md` at Step 0.

| Date | Who | Decision | Why |
|---|---|---|---|
| 2026-11-02 | Claude | Score three value levels: verbatim (exact), strict (BFCL checker), lenient. Headline "AST strict" uses the BFCL checker on every dataset; verbatim is reported alongside it on MASSIVE | The paper says its AST follows the public BFCL, whose checker ignores case, spaces and some punctuation. The brief defined MASSIVE strict as verbatim, which would make our number harsher than the paper's. Both are kept |
| 2026-11-02 | Claude | `[]` on a call item earns no format reward; plain text on a no-call item earns 0.9 | Otherwise replying `[]` every time collects easy reward |
| 2026-11-02 | Claude | Lenient normaliser maps number words in English only; digits in any script | Word tables for 10 languages aren't worth the build. Stated in the write-up |
| 2026-11-02 | Claude | MLCL language mismatch reports presence only; Igbo uses copied-from-query instead of script | MLCL's three subtypes need manual labels. Stated in the write-up |
