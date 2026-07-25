#!/bin/bash
#
# Submits run_psp_pipeline.sh (the FFHQ ALAE -> pSp data pipeline) as an
# LSF cluster job. Mirrors the bsub job-submission pattern used elsewhere
# in this project (e.g. ContactUBS/mouse_single_experiment.sh).
#
# Usage:
#   ./submit_psp_pipeline_job.sh              # full run
#   ./submit_psp_pipeline_job.sh --limit 200  # smoke test on the cluster
#
# Tune WALLTIME/MEM below once you know how big data/latents.npy actually
# is - full FFHQ-scale decode+encode can take a while and use a lot of RAM
# to hold the intermediate image array, the 5:00/5000MB defaults used for
# plain training jobs in this repo are not enough here.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WALLTIME="24:00"
NCPUS=2
MEM=32000

output_dir="job_outputs"
mkdir -p "$output_dir"

job_name="ffhq_psp_data_pipeline"
script_file="${output_dir}/${job_name}_job.bsub"

pipeline_args="$*"

{
  echo "#!/bin/bash -l"
  echo ""
  echo "## Scheduler parameters ##"
  echo "#BSUB -J ${job_name}"
  echo "#BSUB -o ${output_dir}/${job_name}.%J.stdout"
  echo "#BSUB -e ${output_dir}/${job_name}.%J.stderr"
  echo "#BSUB -W ${WALLTIME}"
  echo "#BSUB -n ${NCPUS}"
  echo "#BSUB -M ${MEM}"
  echo '#BSUB -gpu "num=1"'
  echo "module load cuda/12.6"
  echo ""
  echo "## Job parameters ##"
  echo "./run_psp_pipeline.sh ${pipeline_args}"
} > "$script_file"

bsub < "$script_file"
echo "Submitted job: $job_name"
