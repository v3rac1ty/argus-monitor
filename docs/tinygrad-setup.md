# tinygrad training setup

This records the pinned local setup used for GPU-accelerated training, so it can be
reproduced if the machine is reimaged or a teammate needs to set up the same environment.
Nothing in `training/` that ships to the printer depends on this — see the note at the
bottom.

## Checkout

tinygrad is not installed from PyPI. It is a local, editable checkout pinned to a
specific commit:

- Path: `/Users/rishivemulapalli/tinygrad`
- Branch: `tinygpu`
- Commit: `33cd373ad35371ccb483c9645d0c0637a04debc2`
- Installed into the Anaconda environment (`/opt/anaconda3`) as an editable install
  (`pip install -e .` from that checkout). `pip show tinygrad` reports
  `Version: 0.14.0`, `Editable project location: /Users/rishivemulapalli/tinygrad`.

`tinygpu` is a purely local branch name — no branch of that name exists upstream
(`git ls-remote --heads origin tinygpu` returns nothing). The pinned commit itself *is*
still reachable from `origin/master`, verified with
`git merge-base --is-ancestor HEAD origin/master`. The working tree is clean, so the
checkout is exactly this upstream commit with no local modifications.

**Warning:** do not run `git pull` (or `git fetch && git merge`/`rebase`) inside
`/Users/rishivemulapalli/tinygrad` as routine maintenance. tinygrad moves fast and makes
no API-stability promise; advancing the checkout can silently change training behavior
away from what was validated here. If it needs updating, do it deliberately and
re-validate training end to end afterward.

## GPU backend

Training uses a GPU reached over Thunderbolt: an RTX 5060 Ti (16GB), accessed through
tinygrad's NV backend.

**`DEV=NV` is required on every invocation that should use the GPU.** Without it,
`Device.DEFAULT` resolves to `METAL` — the MacBook's own integrated/discrete GPU, not the
5060 Ti — and training will run on the wrong device silently, with no error. There is no
way to tell from the output alone that this happened; the job just runs on the wrong
hardware.

Verify the device before any real training run:

```
DEV=NV /opt/anaconda3/bin/python3 -c "from tinygrad import Tensor; x=Tensor.zeros(1).realize(); print(x.device)"
```

Expected output:

```
NV
```

For comparison, running the same command without `DEV=NV` on this machine prints
`METAL`, confirming the default is the wrong device and the environment variable is load
bearing, not optional.

## Scope: training only

Nothing shipped to the Raspberry Pi depends on tinygrad. The printer-side inference path
uses ONNX Runtime only. tinygrad and the NV/Thunderbolt GPU setup are needed solely for
running or experimenting with training on this development machine.
