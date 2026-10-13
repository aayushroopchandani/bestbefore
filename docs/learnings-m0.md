# What M0 teaches: the ideas behind each step

This is a learning guide, not a plan. For every step of M0 it explains
**what** we did, **why** we did it, and the idea underneath, using real
numbers from our own data. The plan itself is in
[`m0-data-and-test-set.md`](m0-data-and-test-set.md).

It grows as we go: Part A is written. Parts B and C get added when we do
them.

---

## 1. The big picture

### Two different jobs: finding and reading

Reading an expiry date from a photo is really two jobs:

1. **Finding** (called *detection*): *where* in the photo is the date?
   The answer is a box.
2. **Reading** (called *recognition*): *what* does the text inside that
   box say? The answer is `2027-08-15`.

```
   The photo                    Job 1: find               Job 2: read
 ┌────────────────────┐       ┌────────────────────┐
 │  NUTRI DOG FOOD    │       │  NUTRI DOG FOOD    │
 │                    │  ──►  │      ┌──────────┐  │  ──►   2027-08-15
 │  EXP: 15/08/2027   │       │  EXP:│15/08/2027│  │
 │                    │       │      └──────────┘  │
 └────────────────────┘       └────────────────────┘
```

M1 trains a model for job 1 and M3 trains one for job 2. M0 prepares the
data for both.

### Why measure before building

Before you build anything, you measure the tools you already have. That
score is the **baseline**: the number to beat. Without it, you can't say
your new model is better, only that it exists. That's Part C.

---

## 2. Boxes and labels

### What a bounding box is

A **bounding box** is the smallest upright rectangle around something.
It's stored as 4 numbers: the top-left corner `(x1, y1)` and the
bottom-right corner `(x2, y2)`, in pixels. In images, `x` grows to the
right and `y` grows **downward**: `(0, 0)` is the top-left corner.

```
 (0,0) ──────► x
   │
   │      (x1,y1) ┌──────────┐
   ▼              │2021.08.03│
   y              └──────────┘ (x2,y2)
```

A real one from ExpDate's first test photo, which is 1455 × 1092 px:

```
"bbox": [638, 576, 950, 623]
```

That means the date starts 638 px from the left and 576 px from the top,
and ends at 950 px and 623 px. So it's 312 px wide and 47 px tall.

### What labels (annotations) are

A model learns from examples where someone already wrote down the right
answer. Those answers are called **labels** or **annotations**. In
ExpDate, a person drew boxes on every photo and gave each box a
**class**, meaning a category:

| Class | What it is | Example |
|---|---|---|
| `date` | a printed date | `2020.12.12` |
| `exp` | the expiry date (test photos only) | `2021.08.03` |
| `due` | the word that announces an expiry | `EXP`, `Best before`, 까지 |
| `prod` | the word that announces manufacture | `MFG`, 제조 |
| `code` | a batch or lot code | `L22420` |

A model can only learn what the labels show it. Wrong labels teach wrong
things, which is why we look at them first (section 5).

---

## 3. Train, validation and test

### The exam analogy

Think of a student preparing for an exam:

| Data split | Exam analogy | What the model does with it |
|---|---|---|
| **Train** | homework with the answers at the back | learns from it |
| **Validation** | practice tests | we check progress on it and pick settings ("train 50 rounds or 100?") |
| **Test** | the final exam | used **once**, at the end, to get the real score |

If the student sees the final exam questions while studying, they can
memorise the answers and score 100% without understanding anything. A
model does the same: it can memorise photos it has seen. So the test
photos must **never** be used for training or for choosing settings.
This rule is the most important one in machine learning.

### Why we made a validation split

ExpDate gives only train (1,102 real photos) and test (665). But in M1 we
need practice tests to choose settings, and we're not allowed to use the
test photos for that. So we took 15% of the train photos and set them
aside as validation:

```
1,102 real train photos  ──►  937 train  +  165 validation
```

**Why 15%?** People usually pick 10 to 20%. That leaves enough photos
(165) to give a stable score, while keeping most (937) for learning.

### Why "seed 0"

We picked the 165 validation photos **at random**, so that they're a fair
sample, not, say, the first 165 (which might all be similar). But
computers make "random" choices from a starting number called a **seed**.
Same seed, same choices, every time. With seed 0, anyone who reruns our
script gets exactly the same 165 photos, so the results can be
reproduced.

---

## 4. Real and synthetic data

**Synthetic** data is made by a computer: here, dates printed onto
photos in many fonts and styles. **Real** data is photos taken by people.

| | Real | Synthetic |
|---|---|---|
| How many | 1,767 | 11,858 |
| Cost | someone photographs and labels each one | nearly free, unlimited |
| Looks like a store photo? | yes | close, but not quite |

Synthetic data is cheap, but a model trained only on it can fail on real
photos, because the two look slightly different. What we measured:

- Synthetic dates are about **twice as big** in the photo as real ones,
  so they're easier.
- Synthetic photos put the PROD and EXP dates in at random. Some show an
  expiry *before* the manufacture date. So synthetic data teaches the
  model *where* dates are, but not *which* one is the expiry.

That's why M1 trains on **both**, and why we'll measure what synthetic
data adds instead of assuming it helps.

---

## 5. Look at the data before trusting it

### Contact sheets

A **contact sheet** is one big picture made of many small photos with
their boxes drawn on top. We made one for 50 photos from each split, plus
a zoomed-in sheet of just the dates. Thirty seconds of looking catches
problems that no number would show: boxes in the wrong place, sloppy
labels, or a rule nobody wrote down.

### The rule we found: a box covers the date only

Looking at the zoomed sheets, a `date` box covers **only the date value**
(`2021.08.03`), **not** the word before it (`EXP:`) and not a time printed
after it (`08:15`).

This matters for Part B. When you draw boxes on the store photos, you
must follow the same rule. Here's why.

### IoU: how a detector's box gets scored

To decide if a detector "found" the date, we compare its box with the
labelled box using **IoU** (*Intersection over Union*):

```
IoU = area where the two boxes overlap / area covered by either box
```

IoU is 1.0 when the boxes are identical and 0.0 when they don't touch.
The usual rule is: **IoU ≥ 0.5 counts as found.**

Now suppose the detector learned ExpDate's rule and draws a box around the
date only: 100 px wide, 20 px tall. But on the store photo someone
labelled `Best before: 15/08/2027`, a box 230 px wide.

```
overlap = 100 × 20 = 2,000
union   = 230 × 20 = 4,600
IoU     = 2,000 / 4,600 = 0.43   →  below 0.5  →  counted as a MISS
```

The detector was right, but it gets marked wrong, only because the two
datasets were labelled with different rules. Same rule everywhere avoids
that.

---

## 6. Phone photos are often stored sideways (EXIF)

A phone camera sensor always captures in one direction. When you hold
the phone upright, it often **doesn't** turn the pixels. It saves them
sideways and adds a small note inside the file: "rotate 90° when showing
me". That note is part of the **EXIF** data (extra information stored in
a photo, like the date taken and the camera model).

Photo apps read the note, so you never notice. But some code ignores it,
and then:

```
What you see                What code that ignores EXIF sees
┌──────────┐                ┌────────────────┐
│  EXP     │                │                │
│ 15/08/27 │       vs       │  (same photo,  │
│          │                │   on its side) │
└──────────┘                └────────────────┘
```

A box drawn on the upright photo then lands in the wrong place, **with no
error message**. Three ExpDate test photos have this note. We checked:
their boxes were drawn on the upright view.

**What we did:** every script opens photos through one function,
`load_upright`, which always follows the note. One door for all photos
means they can never disagree. Your store photos come from a phone, so
this will matter even more in Part B.

---

## 7. Why we converted to YOLO format

### What YOLO is

**YOLO** ("You Only Look Once") is a popular family of detection models.
It's fast and small, and you can fine-tune it on your own data with a few
commands, using a Python library called **Ultralytics**. That's what M1
will use.

### Why convert at all

Every training library expects its data in **its own format**, the way a
printer expects A4 paper. ExpDate gives us one big JSON file with corner
pixels. Ultralytics' YOLO can't read that. It wants:

1. **One small text file per photo**, with the same name as the photo
   (`test_00001.jpg` → `test_00001.txt`).
2. **One line per box**: `class  centre_x  centre_y  width  height`.
3. **The class as a number** (0, 1, 2, 3), not a word.
4. **Every value between 0 and 1** (a fraction of the photo's size), not
   pixels.

So we translated ExpDate's labels into that format. The information is
the same; only the way it's written down changes.

### A worked example, by hand

The expiry box in `test_00001.jpg` (photo 1455 × 1092 px) is
`[638, 576, 950, 623]`.

```
centre_x = (638 + 950) / 2 / 1455 = 794   / 1455 = 0.545704
centre_y = (576 + 623) / 2 / 1092 = 599.5 / 1092 = 0.548993
width    = (950 − 638)     / 1455 = 312   / 1455 = 0.214433
height   = (623 − 576)     / 1092 = 47    / 1092 = 0.043040
class    = date              = 0
```

And this is the actual line in our file
`data/expdate/yolo/labels/real-test/test_00001.txt`:

```
0 0.545704 0.548993 0.214433 0.043040
```

Same numbers.

### Why fractions instead of pixels?

YOLO **resizes** every photo before looking at it (for example to 640 px
wide). In pixels, the box would have to be recalculated for every size.
As fractions, "the date's centre is 54.6% of the way across" stays true at
any size.

### Why centre and size instead of corners?

Because that's how YOLO itself **predicts** boxes: it guesses a centre
point, then a width and a height. Writing labels the same way means the
label and the model's guess can be compared directly.

### Why numbers for classes?

Models output numbers, not words. A small settings file
(`expdate.yaml`) holds the translation: `0: date, 1: due, 2: prod,
3: code`.

### Why `exp` became plain `date`

The detector's job is only to **find** dates. Working out *which* date is
the expiry is the reader's job, in M3. Also, the training photos have no
`exp` class at all, so a detector couldn't learn it anyway.

### The other files the conversion made

```
data/expdate/yolo/
├── images/        the photos (as shortcuts, see below)
├── labels/        one .txt per photo, same folder layout as images/
├── train_real.txt     list of the 937 real training photos
├── train_synth.txt    list of the 11,858 synthetic photos
├── val.txt            list of the 165 validation photos
├── test.txt           list of the 665 test photos
└── expdate.yaml   tells YOLO where the lists are, and the class names
```

- YOLO finds each photo's label file by swapping `images` for `labels` in
  its path. That's why the two folders mirror each other.
- Real and synthetic photos are in **separate lists**. Training with or
  without synthetic data is then a one-line change in `expdate.yaml`,
  which makes a "does synthetic data help?" experiment easy.

### Shortcuts instead of copies (symlinks)

The YOLO `images/` folder needs the photos in it, but copying them would
use another 1.7 GB of disk. So each "photo" there is a **symlink**: a
tiny shortcut file that points to the real photo, like a shortcut on your
desktop. The catch: the shortcuts store the full path on *this* laptop,
so on the cluster we rerun the converter instead of copying the folder.

### Checking the conversion

A conversion bug is dangerous because it's **silent**. If we had swapped
x and y by mistake, YOLO would train happily on wrong boxes and just get
bad results, with no error. So after writing each label file, the script
reads it back, converts it to corners again, and compares with the
original. The largest difference over all 13,625 photos was **0.003
pixels**, which is just rounding.

---

## 8. Measuring box sizes: will the dates be big enough?

Remember that YOLO shrinks each photo before looking at it. Our real
photos are about 1,200 px on the long side, so shrinking to 640 px makes
everything about half the size.

What we measured:

| | Real photos | Synthetic photos |
|---|---:|---:|
| Date box height at 640 px (median) | **16 px** | 27 px |
| Date box height at 1280 px (median) | 31 px | 53 px |
| Date box share of the photo (median) | 0.45% | 1.3% |

**Why it matters:** a detector struggles when an object is only a few
pixels tall (below about 8 px, it barely sees it). At 16 px our dates are
small but findable. At 1280 px they'd be twice as big, but 1280 px means
4× more pixels, which makes training slower and the model slower on a
phone. So M1 will try both sizes on the validation photos and keep the
better one. That decision comes from the numbers, not a guess.

We also measured **how many date boxes are taller than they are wide**,
a simple sign of sideways text: **0%** in the real photos. ExpDate has
almost no sideways dates, but your store photos do. A model never shown
sideways dates won't learn them, so the M2 generator must make some.

---

## 9. Habits that make results trustworthy

### Fingerprints for files (SHA-256)

A **hash** like SHA-256 turns any file into a 64-character code, its
fingerprint. The same file always gives the same code. Change one byte,
and the code changes completely. We recorded the fingerprint of each
ExpDate zip, so on the cluster the download script can prove it got
exactly the same files. In Part B we use the same trick to "freeze" the
store test set: if anyone changes a label later, the fingerprint won't
match and the code stops.

### Run records

Every script run writes a `run.json` with its settings, the git commit
(which version of the code), the seed, and the library versions. Months
from now, any number in a results table can be traced back to exactly
what produced it, and rerun.

### Automatic tests

`tests/` holds small checks that run in under a second with `pytest`. For
example, one builds a tiny photo marked "rotate 90°" and checks it opens
upright. When we change code later, the tests tell us at once if we broke
something that used to work.

### Paths in `.env`, data outside git

Scripts never hard-code where the data is; they read it from `.env`. The
same code then runs on your laptop and on the cluster, with only `.env`
being different. The data itself never goes into git: it's too big, ExpDate
has its own licence and its own home, and the store photos are private.

---

## Glossary

| Word | Meaning |
|---|---|
| Annotation / label | the correct answer someone wrote down for a training example |
| Baseline | the score of an existing simple method; the number to beat |
| Bounding box | the smallest upright rectangle around an object |
| Class | a category a box belongs to (`date`, `due`, …) |
| Contact sheet | many small photos with their boxes drawn, on one page |
| Detection | finding *where* things are (boxes) |
| EXIF | extra information stored in a photo file, including "rotate me" |
| Fine-tuning | starting from a model already trained on other data and training it a bit more on yours |
| Hash (SHA-256) | a file's fingerprint |
| IoU | overlap of two boxes divided by the area they cover together; ≥ 0.5 usually counts as a match |
| Normalised coordinates | positions as fractions (0 to 1) of the photo's size |
| Recognition | reading *what* the text says |
| Seed | the starting number for "random" choices; same seed, same result |
| Symlink | a shortcut file pointing to another file |
| Synthetic data | training examples made by a computer |
| Train / validation / test | learn from / pick settings on / final score, used once |
| YOLO | a fast family of detection models; Ultralytics is the library that trains it |

---

## Check yourself

<details>
<summary>1. Why can't we use the test photos to choose how long to train?</summary>

Choosing settings by looking at test scores slowly fits the model to
those exact photos, like studying from the final exam. The test score then
looks better than the model really is. That's what the validation split is
for.
</details>

<details>
<summary>2. A box is [100, 50, 300, 90] in a 1000 × 500 photo. What's its YOLO line, if it's a date?</summary>

centre_x = 200 / 1000 = 0.2, centre_y = 70 / 500 = 0.14,
width = 200 / 1000 = 0.2, height = 40 / 500 = 0.08:

`0 0.200000 0.140000 0.200000 0.080000`
</details>

<details>
<summary>3. Why didn't we just copy the photos into the YOLO folder?</summary>

It would use another 1.7 GB for no benefit. Symlinks (shortcuts) give
YOLO the folder layout it expects without a second copy.
</details>

<details>
<summary>4. Your detector finds a date perfectly, but the store label also includes the word "EXP:". Why might it be counted as a miss?</summary>

The labelled box is wider than the detector's box, so their IoU falls.
If it drops below 0.5, the find counts as a miss even though the detector
was right. That's why both datasets must follow the same box rule.
</details>

<details>
<summary>5. ExpDate has 0% sideways dates. What happens if M1 trains only on ExpDate?</summary>

The detector has never seen a sideways date, so it will probably miss
them on the store photos. M2's generator adds sideways dates to fix
exactly that, and the M2 experiment measures whether it worked.
</details>
