1. Change directory to the training source folder `cd training_source`

2. Clone `rrivera1849/LUAR-MUD` inside `training_source`:


```bash
mkdir rrivera1849
cd rrivera1849
git clone https://huggingface.co/rrivera1849/LUAR-MUD
cd ..
```

3. To run on SLURM, you can use: `sbatch ./penn_slurm.sh`. This will create a virtual environment and run training on the HRS 1.3 data in the `data` folder (train on `train` split and validation on the `dev` split). It may need to be customized for your SLURM environment or if running directly on a server you may want to look at `penn_slurm_script.sh` for how to adapt the script for your cluster/environment.

4. This will then produce a `_model` folder inside `./cross_genre_all_output/luar-trainer/` after training is complete. These files inside `_model` can be used to update the files inside the `../docker_container_build_source/datadreamer_lora/checkpoint` folder. The Docker container can then be re-built via the `Dockerfile` in `../docker_container_build_source`.