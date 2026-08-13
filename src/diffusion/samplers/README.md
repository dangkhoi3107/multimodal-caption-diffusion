# Diffusion samplers

Planned owner of reverse/inference update equations such as ancestral DDPM, DDIM and optional ODE solvers.

- It receives a trained prediction model and scheduler/parameterization contracts.
- It returns samples and optional trajectories.
- It does not own training, datasets or model architecture.
- Implementation files are created when Phase 1/A2 starts and must have deterministic fixed-seed tests.
