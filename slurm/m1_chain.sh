#!/bin/bash
# Run by slurm/m1_holder.sbatch once it has the GPU: every M1 training,
# back to back (scripts/m1_chain.py has the order and the decision rules).
set -euo pipefail
cd ~/bestbefore

module load Python/3.11.11      # the venv needs the module's libpython
source .venv/bin/activate
export YOLO_OFFLINE=1           # compute nodes have no internet: never try

python -c "import torch; assert torch.cuda.is_available(), 'PyTorch sees no GPU'; print('GPU:', torch.cuda.get_device_name(0))"
python scripts/m1_chain.py
