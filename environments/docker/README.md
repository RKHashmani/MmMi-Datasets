# Docker Instructions

To build the docker container:

```bash
docker build --platform=linux/amd64 -t rkhashmani/mutual_info_flow_matching:1.0.0 .
```

To interactively test the container:
```bash
docker run -it --platform linux/amd64 --gpus=all --rm=true rkhashmani/mutual_info_flow_matching:1.0.1 /bin/bash
```

Note: If your local setup does not have Nvidia GPUs, you can omit `--gpus=all`.

To deploy:
```bash
docker push rkhashmani/mutual_info_flow_matching:1.0.0
```
