# Running the training
Necessary components:
- checked out code
- dataset in csv_format (I have it in `ml4co-competition/bo_gnn/config/data`, check the `train_gnn.py` file to see where it tries to load from)
- instances in pkl format in dir `ml4co-competition/instances/{problem}/train/`
- singularity file `ml4co-gpu.sif` (optional)
- singularity run rights on euler (optional)

Then you can execute the following on Euler:

```
bsub -n 4 -W 4:00 -R singularity -R "rusage[ngpus_excl_p=1,mem=4096]" singularity exec --bind /cluster/home/rvalentin/Documents/ml4co-competition:/ml4co --pwd /ml4co/bo_gnn/config --nv /cluster/home/rvalentin/singularity-images/ml4co-gpu.sif python train_gnn.py
```
