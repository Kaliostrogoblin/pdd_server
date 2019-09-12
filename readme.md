# Plant Disease Detection server

Repository includes pdd model for the classification. The structure of the client and the server is inspired by this [habr post](https://habr.com/ru/post/445928/).

## Installation steps

### 1. Set up virtual environment for the client

1. Create new virtual environment 
2. Activate it
3. Inside the created venv execute: 

```
pip install -r requirements.txt
```

### 2. Serving model in Docker container

1. Install docker. For details see https://docs.docker.com/install/linux/docker-ce/ubuntu/ 
2. Make script 'docker_run.sh' executable via 

```
sudo chmod +x docker_run.sh
```

3. Run `./docker_run.h` in console, the script will print out id of the created container.

### 3. Get predictions

To get model's predictions, run:

```
python tf_serving_client.py --image $IMAGE_PATH
```

where `$IMAGE_PATH` contains string path to the image for doing inference.