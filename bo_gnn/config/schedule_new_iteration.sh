# $1 = RUN_ID
# $2 = ITERATION

## In the first iteration we don't have tasks to use yet
if [[ $2 -ne 0 ]]; then
    ### LAUNCH DATA GENERATION JOBS ON MULTIPLE CPU NODES
    for i in $(seq 0 9); do
        bsub -J gen_data_job_$i \
            -G ls_krausea -n 20 -W 1:00 -R singularity -R "rusage[mem=4000]" \
            singularity exec --bind /cluster/home/rvalentin/Documents/ml4co-competition:/code,/cluster/project/infk/krause/rvalentin/instances:/instances,/cluster/project/infk/krause/rvalentin/runs:/runs \
            --pwd /code/bo_gnn/config /cluster/home/rvalentin/singularity-images/ml4co-gpu.sif \
            python -m generate_data -r $1 -i $2 -t $i -f train -p one -T 300 -j 20
    done

    ### LAUNCH GNN TRAINING ON GPU JOB, THEN (AFTER FINISHING) LAUNCH TASK CREATOR ON THE SAME NODE
    bsub -w "done(gen_data_job_0)&&done(gen_data_job_1)&&done(gen_data_job_2)&&done(gen_data_job_3)&&done(gen_data_job_4)&&done(gen_data_job_5)&&done(gen_data_job_6)&&done(gen_data_job_7)&&done(gen_data_job_8)&&done(gen_data_job_9)" \
        -J train_gnn_job \
        -G ls_krausea -n 8 -W 0:15 -R singularity -R "rusage[ngpus_excl_p=1,mem=4000]" \
        singularity exec --bind /cluster/home/rvalentin/Documents/ml4co-competition:/ml4co,/cluster/project/infk/krause/rvalentin/instances:/instances,/cluster/project/infk/krause/rvalentin/runs:/runs \
        --pwd /ml4co/bo_gnn/config --nv /cluster/project/infk/krause/rvalentin/singularity-images/ml4co-gpu.sif \
        python train_gnn.py -r $1 -t 150
else
    ### IN THE FIRST ITERATION, WE SAMPLE FROM A RANDOM MODEL
    bsub -J train_gnn_job echo "In iteration 0 we don't need to train yet"
fi

bsub -w "done(train_gnn_job)" \
    -J make_new_tasks_job \
    -G ls_krausea -n 10 -W 0:15 -R singularity -R "rusage[ngpus_excl_p=1,mem=4000]" \
    singularity exec --bind /cluster/home/rvalentin/Documents/ml4co-competition:/ml4co,/cluster/project/infk/krause/rvalentin/instances:/instances,/cluster/project/infk/krause/rvalentin/runs:/runs \
    --pwd /ml4co/bo_gnn/config --nv /cluster/project/infk/krause/rvalentin/singularity-images/ml4co-gpu.sif \
    python make_new_tasks.py -r $1 -i $2 -t 10 -j 20

### AFTER FINISHIN THE PREVIOUS STEP, LAUNCH A NEW ITERATION.
### KILL THIS JOB IF YOU WANT TO CANCEL THE PROGRAM.
bsub -w "done(make_new_tasks_job)" \
    -J schedule_new_iteration_job \
    bash schedule_new_iteration.sh $1 $(($2 + 1))
