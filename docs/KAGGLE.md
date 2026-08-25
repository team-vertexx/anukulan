# Running the GPU backend on Kaggle

Step by step, assuming nothing. About 20 minutes the first time, two minutes
every time after.

Kaggle is the recommended place to develop the CUDA kernels: the GPU quota is
**published** (about 30 hours a week) rather than secret and demand-dependent
the way Colab's is, it needs no payment card, and `nvcc` is already in the
image. It is not the right place for the December demo - see the end.

---

## Before you start

You need a Kaggle account with a **verified phone number**. Without it Kaggle
will not let a notebook use the internet, and the notebook needs the internet to
download the benchmark instances. Verify at kaggle.com/settings under Phone
Verification. This is the single most common thing that goes wrong, and the
error it produces looks like a network failure rather than a permissions one.

---

## Step 1 — Get the code to Kaggle

Two ways. The first is simpler and needs no accounts or tokens; use it for the
first run.

### Option A: upload the code as a private Dataset (simplest)

On your Mac, check the CUDA compiles before you upload anything:

    cd ~/sih26119-solver
    bash scripts/check_cuda_syntax.sh

That stubs the CUDA runtime, strips the `<<<grid, block>>>` launch
configurations, and hands the rest to clang. It catches typos, wrong argument
types and missing members - everything a compiler catches except the parts that
are genuinely about a device. It exists because a round trip to Kaggle was
already spent once on a compile error, and a round trip is twenty minutes.

It must print `cuda_backend.cu type-checks clean`. Then package:

    bash scripts/package_for_kaggle.sh

That writes `sankhya-source.zip`, a few megabytes - source only, no benchmark
data, which gets downloaded on the Kaggle side instead.

Then on kaggle.com:

1. Left sidebar, **Datasets** → **New Dataset**
2. Drag `sankhya-source.zip` in
3. Title it `sankhya-source`
4. Set visibility to **Private**
5. **Create**

Kaggle unzips it for you. The files land at
`/kaggle/input/sankhya-source/`.

### Option B: a private GitHub repository (better once you are iterating)

Push the repo, then in the notebook use a Kaggle Secret holding a GitHub
personal access token:

    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    !git clone https://{token}@github.com/team-vertex/<repo>.git /kaggle/working/sankhya

Worth setting up when you start changing kernels and re-running often, because
updating means a `git pull` instead of a fresh upload.

---

## Step 2 — Create the notebook

1. Left sidebar, **Code** → **New Notebook**
2. Right-hand panel, **Session options**:
   - **Accelerator**: `GPU T4 x2` or `GPU P100` (either is fine)
   - **Internet**: `On`
   - **Persistence**: leave off
3. If you used Option A: **+ Add Input** → **Datasets** → your `sankhya-source`

Check the accelerator actually attached before doing anything else:

    !nvidia-smi

If that prints a table with a GPU name, you are set. If it says command not
found, the accelerator did not attach - fix that first, because everything
below will fail in confusing ways otherwise.

---

## Step 3 — Copy the source somewhere writable

`/kaggle/input` is **read only**. Builds must happen in `/kaggle/working`.

    !cp -r /kaggle/input/sankhya-source /kaggle/working/sankhya
    %cd /kaggle/working/sankhya
    !ls

You should see `CMakeLists.txt`, `src`, `scripts`, `tests`.

---

## Step 4 — Check the toolchain

    !cmake --version && nvcc --version | tail -2

Recent Kaggle images have both. If `cmake` is missing:

    !pip install -q cmake

---

## Step 5 — Run the whole thing

One command does the build, the correctness checks and the timing:

    !bash scripts/gpu_test.sh

It takes 10-20 minutes the first time, most of it downloading benchmark
instances. What it does, in order:

1. confirms a device and `nvcc`
2. builds with `-DSANKHYA_ENABLE_CUDA=ON`
3. runs the backend contract tests - every CUDA operation against the CPU
   reference
4. runs the solver on GPU against **published** Netlib optimal values
5. times CPU against GPU on the large instances
6. crosses CPU/GPU with presolve on/off and writes `results/gpu_matrix.csv`

### What a pass looks like

Step 3 should end with `100% tests passed`. Step 4 prints a table whose
`gpu acc` column is around `1e-06` or smaller on every row, and finishes with
`PASS`.

### What is *not* a failure

The `cpu it` and `gpu it` columns will differ, sometimes by 20% or more. That is
expected and is not a bug. Fused multiply-add alone changes iteration counts
by up to 25% between two CPU builds of the same source, so a different device is
certain to differ. **The gate is the objective against the published optimum**,
which is a number from 1988 that no backend can influence. If the objectives
match and the iteration counts do not, the port is correct.

---

## What this run is meant to settle

Two things went in since the last GPU run, and each has one question attached.

**The fused step-size reduction.** Every iteration the adaptive step rule needs
three inner products, and asking for them one at a time meant three
device-to-host copies - three synchronisations, three pipeline drains. They are
now one kernel, finished on the device, with three doubles crossing the bus.

Where that shows up is a question of iteration count, not problem size:

| instance | iterations | three stalls cost | share of GPU time |
|---|---|---|---|
| `qap15` | ~24,700 | ~1.1 s | **~17%** |
| `datt256_lp` | ~920 | ~41 ms | ~1% |
| `graph40-40` | ~280 | ~13 ms | <1% |

So **`qap15` is the row to watch.** It was 1.11x. If the fusion works it should
move, and `graph40-40` at 1.16x should not - it does not run enough iterations
for reductions to be its problem, whatever else is.

**Presolve.** New, and never run on a GPU. On CPU it is 1.33x geomean over the
18 Netlib instances with published optima. The question is whether it still
pays once the per-iteration cost is a GPU's rather than a CPU's - a smaller
model means fewer nonzeros per kernel, and a GPU that was already underfed may
not care. Step 6 of the script runs every large instance four ways, CPU and GPU
crossed with presolve on and off, and writes `results/gpu_matrix.csv`.

One thing to check rather than assume: presolve reports a `vs original` line
now. If it says `<- over the tolerance`, the reduced model was solved to its
own tolerance but the point that maps back misses the original model's rows.
Re-run that instance with `--abs-tol=1e-8` and it should come out clean.

---

## If graph40-40 stays slow, profile it - do not guess

It is 1.16x, and the reductions are not the reason. Nor is sparsity on its own:
`supportcase10` has 3.35 nonzeros per row against `graph40-40`'s 3.49 and gets
3.89x. So the honest position is that it needs measuring:

    !nsys profile --stats=true -o /kaggle/working/graph40 \
        ./build-cuda/sankhya solve data/lptestset/graph40-40.mps \
            --backend=cuda --tol=1e-4 --quiet

The `cuda_gpu_kern_sum` table in the output says which kernel owns the time. If
it is `spmv_kernel`, the row-to-warp mapping is the thing to change - a whole
32-lane warp per row with 3.5 nonzeros in it uses 11% of the lanes. If the time
is not in kernels at all, it is launch overhead or memory, and that is a
different fix. Bring the table back rather than a conclusion.

---

## Step 6 — The number that matters

Step 5 of the script is the headline. The Netlib instances are too small for a
GPU to pay for itself - expect the GPU to be *slower* there, because a kernel
launch costs more than the whole solve. The large instances are the point:

    !python3 scripts/fetch_lptestset.py --max=4

then re-run `scripts/gpu_test.sh`, or just its last section. Those instances run
to 360,900 rows and 1.5 million nonzeros, which is where a GPU starts to matter.

---

## Getting results back out

Anything written under `/kaggle/working` can be downloaded from the notebook's
**Output** tab after the session ends. Save the tables:

    !cp -r results /kaggle/working/results
    !cp -r bench/results /kaggle/working/results 2>/dev/null || true

The file worth sending back is `results/gpu_matrix.csv`. It has one row per
instance per backend per presolve setting, so the speedups can be recomputed
rather than retyped.

---

## Limits worth knowing

- **30 GPU-hours a week**, and it resets weekly. Kernel work uses very little of
  it: the runs are seconds, the time goes on compiling and downloading.
- **12 hours** maximum per session, and the session dies if the browser tab is
  closed for too long without committing.
- **Commit** (the Save button) runs the notebook top to bottom in the
  background, which is the reliable way to get a long run finished.

---

## Why not Kaggle for the December demo

Free-tier availability is not something to bet a live demo on. For the finale
use **RunPod Secure Cloud**, an RTX 4090 at roughly $0.34-0.69 an hour - under
about 2,500 rupees for the whole 36-hour window. Not Vast.ai: its cheaper
instances are spot instances that can be reclaimed on fifteen seconds notice,
which is a risk not worth the saving.

And record the demo beforehand regardless. Venue networks fail.
