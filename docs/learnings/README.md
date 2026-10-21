# Learnings

Guides that teach the ideas behind each milestone simply, with our own
numbers and links to the lines of code that use each idea. The plans
(what to build, in what order) are one level up, in [`docs/`](..).

| Guide | Covers |
|---|---|
| [`m0.md`](m0.md) | images as numbers, EXIF, resizing and CNN strides, box formats, labels, IoU, train / validation / test, data leakage, domain shift, frozen test sets, failure tags, baselines, right / wrong / none |
| [`m1.md`](m1.md) | convolution, feature maps and YOLO's three grids; neurons, activations, loss, gradient descent, optimisers, warmup, batches and epochs; fine-tuning; augmentation; GPU bottlenecks; how YOLO predicts boxes, the detection loss, NMS, matching, precision, recall, F1, mAP; overfitting and early stopping, leakage, domain shift, seeds, ablations, choosing a threshold by cost; the cluster and Slurm |

Read them in order. In each guide, **★** marks a core idea (read slowly,
do the *Try it* box) and **○** marks one where a quick read is enough.
