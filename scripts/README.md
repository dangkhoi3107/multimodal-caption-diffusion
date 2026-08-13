# Scripts

This directory contains thin command-line entrypoints. A script may parse arguments, load config, construct components, and call application logic from `src/`; it must not contain core preprocessing, model, scheduler, training, sampling, or metric algorithms.

Scripts are added phase by phase as specified in `task.md`.
