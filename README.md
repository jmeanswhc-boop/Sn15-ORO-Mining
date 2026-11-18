# ORO-Tosky

Agent development and research for ORO, Bittensor Subnet 15.

Maintainer: ORO-Tosky <tosky.icdi@outlook.com>.

This repository is a curated workspace snapshot prepared on November 17, 2026.
The original workspace and its analytical data remain preserved locally. The
runtime source is based on [ORO-AI/oro](https://github.com/ORO-AI/oro), commit
`7048ade`; see [the upstream README](README.upstream.md),
[miner guide](docs/miner-guide.md), and [validator guide](docs/validator-guide.md).
Existing licenses and source provenance are retained. Archived third-party
agents retain their original ownership; inclusion does not relicense them.

## Project map

| Directory | Contents |
| --- | --- |
| `agents/` | Tosky and ICDI agent versions, tests, patches and provenance |
| `champions/` | Downloaded champion code and registry metadata |
| `challengers/` | Downloaded publicly released challenger code |
| `race-archives/` | Race rosters, participant records, released sources and results |
| `logs/research/` | Comparative analyses, evidence, collection scripts and verification reports |
| `LV_logs/` | Local validation results and trajectories |
| `problems/` | Existing public problem reference materials |
| `mining-monitor/` | Preserved collector and dashboard source; starting it is optional |
| `src/`, `subnet/`, `docker/` | Upstream agent and validation infrastructure |
| `archive/publication-manifest.json` | Original and published file hashes and sanitization status |

Versioned agents and historical race directories retain their original names,
so existing research references can be followed. Archived sources are reference
material and should not be executed automatically. Snapshot records describe
their collection time; availability and race status may subsequently change.

## Preservation and publication

The publication copy omits credentials, SSH keys, machine configuration, virtual
environments, live monitor databases, and the third-party reference book.
Binary diagnostic bundles remain in the original workspace. Authentication
tokens and signed download URLs in textual evidence are sanitized in this copy;
the manifest records both source and published hashes. Historical checksum
files describe their original collections; use the publication manifest for
this sanitized snapshot. The renamed upstream README is recorded as `README.md`
in the source manifest.

Upstream automation workflows are omitted to prevent accidental image
publication and deployment from this research repository. The dashboard remains
disabled locally; this upload does not start services or run paid verification.

The practice EnvPack was present only as an upstream Git LFS pointer, rather
than its actual payload. Its pointer is preserved under `archive/` for reference;
fetch the signed public practice delivery from the upstream repository before
running local evaluation.

## Development

Consult the upstream guides for dependencies and Docker setup. Configure local
credentials in `.env` using `.env.example`; never commit the populated file.
Public qualifier and practice results are not predictions of hidden race scores.

Each future agent update should preserve its baseline, document its intended
behavioral change, and undergo offline review before paid verification.
