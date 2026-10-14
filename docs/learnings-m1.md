# What M1 teaches: the ideas behind each step

A learning guide to go with the plan,
[`m1-date-detector.md`](m1-date-detector.md). It explains every new idea
the plan uses, in simple words, with our own numbers. A results section
gets added as the slices are done.

---

## 1. What M1 builds

A **detector**: you give it a photo, it gives back boxes, each with a
class and a **confidence** (how sure the model is, from 0 to 1).

```
                         ┌──────────────────────────────┐
  photo of a pack  ───►  │       date detector (M1)     │  ───►  date  0.94  [0.41, 0.62, 0.58, 0.66]
                         └──────────────────────────────┘        due   0.88  [0.33, 0.62, 0.40, 0.66]
                                                                 date  0.21  [0.70, 0.10, 0.80, 0.13]
```

It doesn't read the date; it only says **where** dates are. M3 reads
them.

---

## 2. The cluster: why, and how it's laid out

Training means showing the model about 12,800 photos, many times over.
Your laptop (an M2 with 8 GB of memory) would take a day per training,
and M1 needs about 7 trainings. The cluster's **H100** GPUs are among the
fastest made for this work.

### Login node vs compute nodes

```
  your laptop ──ssh──►  LOGIN NODE (mn1)  ──Slurm──►  COMPUTE NODES
                        - you type here               - cn5, cn6: 2× H100 GPU each
                        - has internet                - do the heavy work
                        - light work only             - NO internet
                        (editing, downloads,          - you don't log into them;
                         submitting jobs)               Slurm runs your job there
```

- The **login node** is shared by everyone who's logged in. Running a
  heavy training there slows it down for all of them, so it's for light
  work only: downloading, setting up, and submitting jobs.
- The **compute nodes** have the GPUs, but **no internet**. So everything
  a job needs (data, Python packages, pretrained weights) is downloaded on
  the login node first. If a job tries to download something, it fails.
  That's why the plan has a smoke test.
- **Storage:** your home folder is the same on all nodes (a shared
  filesystem), with a 950 GB limit. We keep data and checkpoints in
  `~/bestbefore-data/`.

### Slurm: the queue

Many people share 4 GPUs, so you don't run training directly. You **ask**
for resources, and **Slurm** (the scheduler) runs your job when they're
free. It works like a restaurant waiting list: you give your name and
party size, and you're seated when a table opens.

The words:

| Word | Meaning |
|---|---|
| **job** | one piece of work you hand to Slurm |
| **`sbatch script.sbatch`** | submit a job to run in the background; you can log out |
| **`srun`** | run one command on a compute node and wait for it |
| **`squeue -u $USER`** | see your jobs: waiting (`PD`) or running (`R`) |
| **`scancel <id>`** | cancel a job |
| **partition** | a group of machines with rules; ours is `gpu` (3-day limit) |
| **`--gres=gpu:1`** | "I need 1 GPU" (gres means *generic resource*) |
| **job array** | many similar jobs from one script, like 4 trainings with different settings |
| **QOS limit** | our account may run **2 jobs at a time**; extra ones wait |

A job script is a normal shell script with `#SBATCH` lines at the top
that tell Slurm what you need:

```bash
#!/bin/bash
#SBATCH --job-name=m1-train            # a name you'll see in squeue
#SBATCH --partition=gpu                # the GPU machines
#SBATCH --gres=gpu:1                   # one H100
#SBATCH --cpus-per-task=16             # CPUs to load and resize photos
#SBATCH --mem=64G                      # memory
#SBATCH --time=24:00:00                # stop after 24 h no matter what
#SBATCH --output=slurm/logs/%x-%j.out  # where printed output goes (%j = job id)

module load Python/3.11.11             # the cluster's Python
source ~/bestbefore/.venv/bin/activate # our packages
cd ~/bestbefore
python scripts/train_detector.py "$@"  # "$@" passes on your settings
```

---

## 3. Fine-tuning: not starting from zero

A brand-new model knows nothing. It would need a huge number of photos to
learn even basic things, like what an edge or a letter looks like. So we
start from **pretrained weights**: a YOLO model already trained by
Ultralytics on COCO, a big dataset of everyday objects (people, cars,
cups). It already "knows how to see".

**Fine-tuning** means continuing to train that model on our photos, so it
learns one new thing: printed dates. It's like a person who already
reads English learning to read a new font. Much faster than learning to
read from scratch.

---

## 4. How training works

### Epochs and batches

- A **batch** is a small group of photos the model looks at together (we
  use 16). After each batch, it adjusts itself a little.
- An **epoch** is one full pass through all the training photos. With
  about 12,800 photos and batches of 16, one epoch is about 800 small
  adjustments.
- We train for up to **100 epochs**.

### Loss: how wrong the model is

After each batch, the model's boxes are compared with the labels. The
**loss** is one number saying how wrong they were: boxes in the wrong
place, missed dates, wrong classes. Training is just "change the model a
little, in the direction that makes the loss smaller", repeated many
thousands of times.

### Validation after every epoch, and early stopping

After each epoch, the model is scored on the **165 validation photos** it
never trains on. Two things can happen:

```
score on validation
  ▲
  │          ●  ●  ●  ●  ●            ← stops improving: stop here
  │       ●                ●  ●       ← getting worse: overfitting
  │    ●
  │ ●
  └────────────────────────────────►  epochs
```

- At first the score rises: the model is learning.
- Later it can **fall**: the model has started memorising the training
  photos instead of learning what a date looks like. That's
  **overfitting**, like a student who memorises past papers word for word
  and fails new questions.

**Early stopping** watches this. If the validation score hasn't improved
for 30 epochs (the **patience**), training stops and the best epoch's
model, `best.pt`, is kept.

---

## 5. The two settings M1 compares

### Model size: `n` or `s`

A model is made of **parameters**: numbers adjusted during training.
More parameters can learn more detail, but run slower.

| Model | Parameters | Trade-off |
|---|---:|---|
| `yolo11n` (nano) | 2.6 million | fastest, smallest |
| `yolo11s` (small) | 9.4 million | more detail, about 3× slower |

### Input size: 640 or 1280 px

YOLO shrinks every photo to a fixed size before looking at it. In M0 we
measured that a real date is about **16 px** tall at 640 and **31 px** at
1280. Bigger is easier to see, but 1280 is **4×** the pixels, so it's
slower to train and to run.

We don't guess which is best: we train all four combinations and compare
them on validation. That's experiment C1.

---

## 6. Augmentation, and why we turn off flipping

**Augmentation** means showing the model slightly changed copies of each
training photo: brighter, darker, zoomed, shifted, or stitched together
with other photos (YOLO's "mosaic"). The model sees more variety without
anyone labelling more photos, so it copes better with new ones.

By default YOLO also **flips photos left-to-right** half the time. For
cars or dogs that's realistic: a car facing left is still a car. For text
it isn't:

```
original:  15/08/2027
flipped:   7202\80\51     (mirror image: never printed on any pack)
```

Training on mirrored dates wastes effort on something that never
happens, so we turn flipping off. We also leave rotation off: in M2 we
want to measure exactly what sideways synthetic dates add, and rotation
here would blur that comparison.

---

## 7. Confidence, threshold, precision and recall

Every box comes with a **confidence**. We choose a **threshold**: keep
boxes above it, throw away the rest. The threshold controls a trade-off
between two scores:

- **Recall:** of all the real dates, how many did we find?
- **Precision:** of all the boxes we kept, how many were real dates?

**Example.** 10 photos with 10 real dates.

| Threshold | Boxes kept | Correct | Recall | Precision |
|---|---:|---:|---:|---:|
| low (0.2) | 12 | 8 | 8 / 10 = 0.80 | 8 / 12 = 0.67 |
| high (0.6) | 9 | 7 | 7 / 10 = 0.70 | 7 / 9 = 0.78 |

A low threshold finds more dates but also keeps more junk. A high one is
cleaner but misses more. **F1** combines the two into one number:

```
F1 = 2 × precision × recall / (precision + recall)

low:   2 × 0.67 × 0.80 / (0.67 + 0.80) = 0.73
high:  2 × 0.78 × 0.70 / (0.78 + 0.70) = 0.74
```

We pick the threshold with the best F1 **on validation**, then use that
same threshold everywhere else. Choosing it on the test photos would be
cheating (section 9).

For this project, **recall matters most**: a date the detector misses
can never be read. A few extra boxes only cost the reader a bit of time.

---

## 8. mAP@0.5: one number for the whole trade-off

Recall and precision depend on the threshold. **mAP@0.5** scores the
detector at *every* threshold at once:

1. Slide the threshold from high to low. At each step, note recall and
   precision. That traces a **precision-recall curve**.
2. **AP** (average precision) is the area under that curve. 1.0 means
   every date found with no junk at any threshold; lower is worse.
3. **mAP** is the mean of AP over the classes (`date`, `due`, `prod`,
   `code`). We report the `date` one as the headline.
4. **@0.5** means a box counts as correct when its IoU with a label is at
   least 0.5 (the same IoU as in [`learnings-m0.md`](learnings-m0.md)).

So mAP@0.5 answers "how good is this detector overall", while recall at
our threshold answers "how many dates will it actually find in use". The
plan reports both.

---

## 9. Making the results trustworthy

### Decide on validation, test once

The plan answers its two questions (which size, and does synthetic data
help) on the **validation** photos, with decision rules written down
**before** the runs. Only after every choice is fixed are the **test**
photos used, once. If we kept checking test scores while choosing, we'd
slowly pick whatever happens to suit those 665 photos, and the final
number would flatter the model.

### Why train 3 times (seeds)

Training has randomness in it: the order of photos, the augmentations,
the starting point of the new layers. Train twice with different
**seeds** and you get slightly different models and scores. With one run,
you can't tell whether a 1-point difference is real or luck. With 3, we
report the **mean and the range**, and the reader can see how much the
number wobbles. We decide in advance that seed 0's model is the official
one, so we can't pick the luckiest.

### Why "within 1 point, pick the cheaper one"

165 validation photos is not many: one or two photos going differently
can move mAP by about a point. So a difference that small doesn't prove
anything, and the plan then prefers the smaller, faster setup.

### Why the store photos stay on the laptop

The store photos belong to the retailer, and the rule is that only
summary numbers leave the laptop. The cluster is another machine, so the
photos can't go there. Instead, the trained model, a file of a few MB,
comes **to** the laptop, and the store photos are scored here.

---

## 10. Moving code and files around

- **SSH key:** a secure login without typing the password. You make a
  pair of files: a private key that stays on your laptop, and a public
  key you put on the cluster. The cluster checks that they match.
- **`~/.ssh/config` alias:** lets you type `ssh bbhpc` instead of the
  full username and address. It lives only on your laptop, never in the
  repo.
- **`git clone` on the cluster:** the cluster gets the code from GitHub,
  so every training run can record exactly which commit it used.
- **`rsync`:** copies files between the laptop and the cluster, such as
  results and trained models, sending only what changed.
- **One direction for each:** code goes laptop → GitHub → cluster;
  results come cluster → laptop; all commits happen on the laptop.

---

## What slice A1 taught us (2026-10-14)

Things that went wrong or surprised us while setting up, and why:

- **The SSH key needed the password exactly once.** `ssh-keygen` made the
  key pair on the laptop; `ssh-copy-id` logged in with the password one
  last time to put the public key on the cluster. Since then,
  `ssh bbhpc` logs straight in, which is also what lets Claude run
  cluster commands.
- **The cluster's Python module has `python3`, not `python`.** Typing
  `python` gave "command not found". Clusters often only provide the
  `python3` name; inside our venv, both names work.
- **`module load` is needed every time, not just once.** The venv's
  Python is a link to the module's Python, which needs a shared library
  (`libpython3.11.so`) that only appears after `module load
  Python/3.11.11`. Without it: "error while loading shared libraries".
  That's why the job script starts with `module load`.
- **A `module` is how clusters offer software.** Many users need
  different versions of Python, CUDA and so on, so nothing is installed
  "globally". `module load X` switches version X on for your current
  shell only, by changing where the shell looks for programs and
  libraries.
- **PyTorch is big.** The install downloads GPU libraries (CUDA, cuDNN)
  of several GB, so it takes many minutes even on a fast network. It
  runs on the login node because that's the only node with internet.
- **The GPU driver decides which PyTorch you can use.** Three layers have
  to agree:

  ```
  PyTorch build  ──needs──►  CUDA version  ──needs──►  GPU driver version
  torch 2.14 (default)        CUDA 13.0                  580 or newer
  torch 2.11.0+cu128          CUDA 12.8                  570 or newer   ← our cluster: 570.86
  ```

  `pip install torch` picked the newest build, made for CUDA 13. Our
  cluster's driver is 570.86, too old for CUDA 13, so PyTorch would have
  installed fine and then quietly reported "no GPU" inside every job. We
  caught it before running anything, by finding the driver version on
  the login node (the file `libcuda.so.570.86.10`), and installed the
  CUDA 12.8 build instead. Lesson: on a shared cluster you can't upgrade
  the driver, so you pick the PyTorch that fits the driver.
- **The laptop gets the same libraries.** It's needed for the laptop
  smoke test and for scoring the store photos, which never leave the
  laptop. On the Mac, PyTorch uses Apple's GPU through **MPS** (Metal
  Performance Shaders) instead of CUDA.

---

## Glossary

| Word | Meaning |
|---|---|
| Augmentation | training on changed copies of photos (brighter, zoomed, …) |
| Batch | the group of photos processed together before each adjustment |
| Checkpoint / weights | the saved model file, e.g. `best.pt` |
| Compute node | a cluster machine that runs jobs; ours have GPUs and no internet |
| Confidence | how sure the model is about a box, 0 to 1 |
| Early stopping | ending training when validation stops improving |
| Epoch | one pass through all the training photos |
| F1 | one score combining precision and recall |
| Fine-tuning | continuing to train a pretrained model on your own data |
| GPU | a chip that does the maths of training many times faster than a CPU |
| Login node | the cluster machine you log into; light work only |
| Loss | one number saying how wrong the model's boxes were |
| mAP@0.5 | the area under the precision-recall curve, averaged over classes, with IoU ≥ 0.5 counting as correct |
| Module | a cluster's way of switching a software version on for your shell (`module load Python/3.11.11`) |
| MPS | Apple's GPU backend for PyTorch on Macs, used instead of CUDA |
| Overfitting | memorising the training photos instead of learning the general idea |
| Parameters | the numbers inside a model that training adjusts |
| Patience | how many epochs without improvement before early stopping |
| Precision | of the boxes kept, the share that are real dates |
| Pretrained weights | a model already trained on a big general dataset |
| Recall | of the real dates, the share that were found |
| Slurm | the program that queues jobs and runs them on compute nodes |
| Threshold | the confidence below which boxes are thrown away |

---

## Check yourself

<details>
<summary>1. Why can't a training job download YOLO's pretrained weights by itself?</summary>

Jobs run on compute nodes, which have no internet. The weights must be
downloaded on the login node first, into a folder the job then reads.
</details>

<details>
<summary>2. Validation mAP@0.5 is 0.912 for s640 and 0.918 for s1280. Which does the plan pick, and why?</summary>

s640. The difference (0.6 points) is under the 1-point rule, so it's
treated as noise, and the cheaper setup wins: same model size, smaller
input.
</details>

<details>
<summary>3. At some threshold the detector keeps 20 boxes and 15 of them are real dates, out of 18 real dates. What are precision and recall?</summary>

Precision = 15 / 20 = 0.75. Recall = 15 / 18 = 0.83.
</details>

<details>
<summary>4. Why is horizontal flipping turned off?</summary>

A flipped date is mirrored text, which never appears on a real pack.
Training on it wastes effort and can teach the model wrong shapes.
</details>

<details>
<summary>5. Why not choose the input size by looking at how small the dates are in the store photos?</summary>

That uses the test set to make a choice, which is a form of tuning on the
test. The store result would then look better than it should. The input
size is chosen on ExpDate's validation split only.
</details>
