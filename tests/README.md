# Tests

Tests are added with each vertical slice. The suite covers mathematical contracts, preprocessing edge cases, tensor shapes, masks, finite gradients, deterministic sampling, decoding, and small smoke tests.

Long full-training runs do not belong in the unit test suite. One-image and mini-batch overfit experiments should use dedicated short experiment commands and save evidence under `outputs/`.
