# `scripts/rwt.py` — release smoke tester

`make test` (`scripts/test.py`) tests the tree you just built. `rwt.py` tests
the artifact users actually download: it fetches a release archive from GitHub,
unpacks the binary into `build/rwt/`, and drives that binary's CLI across every
modality — embeddings, transcription, generation, vector store, image.

It is the successor to the shell scripts that lived in `scripts/case/`; every
one of those is now a case in the suite, so they share one model registry, one
timeout, one summary, and one place that knows a GPU build needs
`--gpu-layers`.

## The three commands

```sh
python3 scripts/rwt.py install --cuda        # latest linux-x86_64-cuda release -> build/rwt/
python3 scripts/rwt.py test --cuda test-all  # the whole matrix
python3 scripts/rwt.py clean                 # remove the binary + everything it wrote
```

`run` is all three in one, stopping at the first failure:

```sh
python3 scripts/rwt.py run --cuda            # install, test-all, clean
python3 scripts/rwt.py run --cuda --fast     # embed-1, gen-1, sd-4 instead of test-all
```

A failing `run` deliberately leaves the binary in place — the thing worth
inspecting when a release fails is the binary it failed with.

## Picking what to install

`--cpu` / `--metal` / `--cuda` / `--vulkan` / `--rocm` / `--sycl` name a
backend; the host's OS and architecture pick the asset from there
(`--cuda` on Linux → `chimera-<version>-linux-x86_64-cuda.tar.gz`). Only the
pairs CI actually publishes are known — `python3 scripts/rwt.py list assets`
prints them for this machine, and asking for one that does not exist fails
before anything is downloaded rather than on a 404.

`--version 0.2.16` pins a release; without it, `/releases/latest` decides.
`--asset` overrides the choice entirely and takes a local archive, a full URL,
or a bare asset name:

```sh
python3 scripts/rwt.py install --asset dist/chimera-0.2.16-linux-x86_64-cuda.tar.gz
python3 scripts/rwt.py install --asset https://github.com/shakfu/chimera/releases/download/0.2.16/chimera-0.2.16-linux-x86_64-cuda.tar.gz
```

To test something that is not a release artifact at all, skip `install` and
point the suite at the binary:

```sh
python3 scripts/rwt.py test --bin build/chimera test-all
```

`clean` never deletes a binary named with `--bin`.

## Test targets

`python3 scripts/rwt.py list tests` prints them. One token per test, matching
the rules `gen-makefile` emits:

| family | cases |
|--------|-------|
| `embed` | multi-text vectors via `--embd-separator`; corpus file → one vector per line, memoized |
| `transcribe` | `jfk.wav` speech-to-text; the same with `--output-{srt,vtt,json}` files |
| `gen` | Llama-3.2-1B; Qwen3-4B; Gemma-4-E4B with sampler knobs |
| `rag` | `index create`/`ingest`/`search` via `$CHIMERA_DB`; the same with an explicit `--db`, all three retrieval modes, `stats`, `db status`, `drop` |
| `sd` | Z-Image Turbo: baseline, flash-attn, cfg-1 + flash-attn, cfg-1 + offload + flash-attn |

`test-<family>-all` runs one family, `test-all` runs everything. Every case
reports its own exit code and wall time in the summary; a missing model is a
skip (rc 2), not a failure.

Two things the suite knows that the bare CLI does not:

- **`--gpu-layers` is not optional.** chimera's llama-side subcommands default
  to 0, so a GPU release tested without it would pass every case while
  measuring nothing but the CPU path. The backend picks the default (99 on a
  GPU backend, 0 on `cpu`); `--gpu-layers N` overrides it.

- **Nothing touches the user's database.** `--db` and `--cache-db` are passed
  explicitly, or `$CHIMERA_DB` is pointed into the output directory, so a smoke
  test never creates collections in the real store.

## Where things go

The script's entire footprint is one directory, `build/rwt/`:

```
build/rwt/chimera      the installed binary
build/rwt/downloads/   the release archives it was unpacked from
build/rwt/out/         images, transcripts, scratch DBs
```

`downloads/` is a sibling of `out/` rather than a child: an archive is the
*input* to a run — the artifact under test — while `out/` is what a run
produced.

Under `build/` because it is already gitignored — a smoke run leaves nothing for
the next `git status` to report — and already what `make clean` sweeps, so a
tested binary is thrown away along with the build tree it was checked against.
One directory under that, so what rwt.py owns inside a tree it shares with cmake
is obvious at a glance.

`--build-dir` moves the base (it also honours `BUILD_DIR`, the same variable the
Makefile reads, so an out-of-tree build directory is named once and both agree
on it). `--bin-dir` and `--out-dir` override the binary and the output directory
individually, and win over `--build-dir` when both are given. `clean` removes
the installed binary, the downloads and the output directory, then `build/rwt/`
itself once nothing is left in it.

`models/` is the deliberate exception: it stays at the project root, shared with
`make test`, because re-downloading tens of GiB of weights after every
`make clean` is not a sensible default.

## Models

`python3 scripts/rwt.py list models` shows what is needed and what is already
on disk; `download all` fetches the lot. Each source can be redirected with
`CHIMERA_MODEL_<KEY>=<repo_id>:<filename>` — needed for `gemma-e4b` and
`z-image-turbo`, whose default URLs are best-effort. `jfk.wav` comes from the
vendored whisper.cpp tree when `make deps` has run and is downloaded otherwise;
the RAG corpus is the checkout's own `README.md` + `CHANGELOG.md`, falling back
to a small generated corpus when the script is run standalone.

## Standalone use

The script has no dependencies beyond the standard library (`huggingface_hub`
only if a model has to be fetched by repo id rather than URL), and finds its
own root by walking up to the nearest `CMakeLists.txt` or `.git`. Copied into
an empty directory it will create `models/` and `build/rwt/` there, which is
the intended way to check a release on a machine that has no checkout.

## Regression coverage

The rag cases deliberately create their collections at chimera's *default*
`--chunk-tokens 512`. That is the configuration that used to abort
`index ingest` outright:

```
llama-context.cpp: GGML_ASSERT(cparams.n_ubatch >= n_tokens &&
                               "encoder requires n_ubatch >= n_tokens") failed
```

`Embedder::embed()` now clamps an over-long input instead of letting
`llama_context` abort the process, so the default path works. Leaving the cases
on the defaults is what would catch that regression coming back — the suite
exists to test what users get, not a configuration chosen to avoid a bug.
