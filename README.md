# Swiss Alps Treeline Change

This repository contains the initial analysis code for the **Swiss Alps treeline change project**. The project aims to quantify changes in treeline elevation between historical and contemporary observations and to explore the potential environmental drivers of those changes.

## Current contents

- `treeline_shift_quantification.py`: performs the basic treeline-shift analysis. It traces historical treeline sample points toward the contemporary treeline using a high-resolution digital elevation model, calculates elevation changes, applies quality checks, and produces summary outputs.
- `Figure5_RF.py`: uses a random forest model to investigate potential drivers of treeline elevation change and calculate permutation-based variable importance.

## Project status

This repository is at an early stage of development. The code, configuration, paths, comments, and documentation have not yet been fully organized. The project will be cleaned, documented, and expanded progressively.

Input data are not included in this repository. Placeholder paths such as `xx` must be replaced with local data paths before running the scripts.
