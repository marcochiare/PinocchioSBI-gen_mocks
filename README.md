# PinocchioSBI: generate mock catalogues of galaxy clusters

Cluster cosmology using simulation-based inference (SBI) with Pinocchio.

--- 

This repository is used to:

1. generate a Sobol sequence that combines different cosmological parameters with `gen_mocks_sobol.py` (avail. $\Omega_m$, $\sigma_8$, $h$, $w_0$, $w_a$).
2. setup a dataset of Pinocchio runs.
3. run the Pinocchios and monitor their execution.
4. post-process the dataset to "paint" clusters on the halo catalogues of Pinocchio, and obtain cluster observables (lensing, richness, number counts ...). This is obtained through the CosmoPostProcess package. The two modules here provided are still in active development.

This is achieved through the bash jobs in `jobs/` (mainly `run_setup_pinocchio.sh`, `run_pinocchio.sh`, `run_zshells.sh`, `submit_postprocess_painting.sh` and `submit_postprocess_richness.sh`), which require minimal changes (modules and environments mainly); it is thought to be as straight-forward as possible. The use of bash environment variables is advised. E.g.:

```console
MAIN_DIR="/absolute/path/to/dir"
EXEC="/absolute/path/to/exec"
export MAIN_DIR
export EXEC
sbatch run_pinocchio.sh
```

The scripts are meant and tested for HPC clusters using SLURM.





