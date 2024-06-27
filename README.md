# PyTorch STORM Optimizer

Unofficial PyTorch implementation of STORM optimizer in the paper [Momentum-Based Variance Reduction in Non-Convex SGD](http://papers.neurips.cc/paper/9659-momentum-based-variance-reduction-in-non-convex-sgd.pdf).

This implementation is based on the official implementation in [google-research/storm](https://github.com/google-research/google-research/tree/d173c826ec2542a8d270054a25291608a9203f15/storm_optimizer).

## Remarks

- Existing PyTorch implementations of STORM optimizer are not correct, they did not correctly use gradients of current and previous models on same batch.
- This implementation does compute [two gradients](https://github.com/google-research/google-research/blob/d173c826ec2542a8d270054a25291608a9203f15/storm_optimizer/storm_optimizer.py#L152) on the same batch as original works in TensorFlow.
- This implementation utlize a cashed future batch to compute and store the gradients of the future batch, getting rid of the need to load and recover the model parameters when computing the future gradients.
- This implementation is use exactly the same paradigm as the original TensorFlow implementation, including the [gradient clamp](https://github.com/google-research/google-research/blob/d173c826ec2542a8d270054a25291608a9203f15/storm_optimizer/storm_optimizer.py#L179) and the [momentum trimming](https://github.com/google-research/google-research/blob/d173c826ec2542a8d270054a25291608a9203f15/storm_optimizer/storm_optimizer.py#L173). 

