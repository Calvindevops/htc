"""Wiki: LLM-synthesized knowledge pages that complement the raw memory.

`build_wiki` derives topics from the memory (or accepts explicit ones) and
synthesizes one grounded, cited page per topic. `add_wiki_to_memory` writes
those pages back into the memory as `kind="wiki"` chunks, so retrieval hits
both raw source chunks and synthesized pages. `write_wiki_files` writes them
to `<root>/.htc/wiki/` for human browsing.
"""

from __future__ import annotations

from .generator import UNKNOWN, WikiPage, add_wiki_to_memory, build_wiki, write_wiki_files

__all__ = [
    "UNKNOWN",
    "WikiPage",
    "add_wiki_to_memory",
    "build_wiki",
    "write_wiki_files",
]
