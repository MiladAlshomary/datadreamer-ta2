This folder contains instructions on how to reproduce PAUSIT's System #2 training from the HIATUS 1.3 eval.

- Training instructions can be found in `training_source/README.md`.
- Docker container building instructions can be found in `./docker_container_build_source/` and the container can be built with the `Dockerfile`.
- Once trained, `training_source/README.md` also contains instructions how to update the model checkpoint folder inside the `../docker_container_build_source/datadreamer_lora/checkpoint` folder, so the container can be re-built.