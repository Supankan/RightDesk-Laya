# Third-party data and model provenance

RightDesk keeps datasets and weights out of Git. Download commands are explicit and
record their pinned revision, source, license metadata and SHA-256 checksums locally.

| Resource | Pin | Recorded license | Use |
|---|---|---|---|
| [ABCD dataset](https://github.com/asappresearch/abcd) | `6b8700ce67c6b37b062dd7a60abc76d7ef832a97` | MIT in the source repository | Original first customer messages; separate scenario-derived eventual-owner references |
| [Laya checkpoint](https://huggingface.co/convaiinnovations/laya) | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | Apache-2.0 in the model card | Local CPU inference, six queue choices |

The ABCD download stores the source license with the archive. The model download
keeps the model card and an integrity manifest. Retain those notices when handling
or redistributing the corresponding resources. Python dependency versions are in
`uv.lock`; their own packages contain their respective license notices.

Prepared references are provisional research labels. They do not assert that the
first message contains enough information for its eventual queue. Human judgments
have separate provenance and append-only reviewer revisions.
