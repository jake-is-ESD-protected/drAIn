# drAIn
BS-less, platform-agnostic ML training and inference pipeline for TensorFlow models.

## Install

### System-wide or virtual environment
```bash
pip install .[full] # full version for training
pip install .[lite] # reduced version for inference with TFLite
```

### Docker

**Building:**
```bash
docker build -t drain:full -f Dockerflies/Dockerfile.full . # full version for training 
docker build -t drain:lite -f Dockerflies/Dockerfile.lite . # reduced version for inference with TFLite
```
**Running:**
```bash
docker run \
    --gpus all \
    -it \
    --rm \
    --runtime nvidia \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    -v .:/workdir \
    drain:full

docker run \
    -it \
    --rm \
    --ipc=host \
    --network=host \
    -v .:/workdir \
    drain:lite
```