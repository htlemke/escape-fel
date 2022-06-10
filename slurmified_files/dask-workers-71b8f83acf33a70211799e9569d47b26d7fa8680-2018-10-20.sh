#!/bin/bash

#SBATCH -e logs/dask-workers-71b8f83acf33a70211799e9569d47b26d7fa8680-2018-10-20.%J.err
#SBATCH -o logs/dask-workers-71b8f83acf33a70211799e9569d47b26d7fa8680-2018-10-20.%J.out
#SBATCH -J dask-workers-71b8f83acf33a70211799e9569d47b26d7fa8680-2018-10-20

#SBATCH --mem-per-cpu=100
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19
#SBATCH --cpus-per-task=1

set -eo pipefail -o nounset


###

/sf/bernina/anaconda/ahl/bin/dask-worker --nthreads 1 --nprocs 1 --reconnect --nanny --bokeh  --local-directory "/photonics/home/lemke_h/mypy/escape-fel/slurmified_files" sf-cn-1.psi.ch:34628