# Local development corpus

`dev_corpus.txt` is original English prose generated locally for BitForge's
Milestone 3 by the coding assistant. It is not copied from a book or downloaded
dataset. It describes a fictional workshop; no real personal data is included.
The generated text is offered under CC0-1.0 (public-domain dedication, to the
extent any rights apply). There are no third-party source passages.

The exact UTF-8 bytes, SHA-256, character/paragraph counts, and provenance are
recorded in `dev_corpus.manifest.json`. The corpus is marked `-text` in
`.gitattributes` to prevent newline conversion during Git checkout. Preserve LF
newlines when reproducing the hash. Training records its actual input hash and rejects a mismatch with
the bundled manifest. Alternate local corpus files are accepted explicitly,
with their actual hashes recorded; their permissions are the user's responsibility.

This is a small, repetitive, assistant-generated development fixture, not a
representative natural-language benchmark. Splits are contiguous and made
before tokenization/window generation. Shared characters, phrases, and topics
between splits are expected; no held-out corpus was used to tune this text.
