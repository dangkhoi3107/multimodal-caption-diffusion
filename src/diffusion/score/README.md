# Score-based diffusion

Planned owner of denoising score matching, annealed Langevin dynamics, SDE definitions and continuous-time samplers.

- A4 starts with a 2D MLP score model.
- A5 adds reverse SDE and probability-flow ODE contracts.
- This package must keep score scaling explicit instead of treating epsilon prediction and score as identical tensors.
- No implementation is present until the corresponding checkpoint starts.
