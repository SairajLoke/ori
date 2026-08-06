docker stop origami-contract-policy 

docker rm origami-contract-policy 

docker build -t "$IMAGE" .

docker run -d --name origami-contract-policy \
  --gpus all \
  --network origami-contract-test \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges=true \
  --user 65532:65532 \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=4g \
  --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
  --shm-size 2g \
  --memory 10g \
  --cpus 4 \
  --pids-limit 512 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-contract-router:7447 \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  "$IMAGE"

# docker restart origami-contract-policy
#    

# export TORCHINDUCTOR_CACHE_DIR=/home/sai/.cache/torchinductor



# no constrained run 
# docker run -d --name origami-contract-policy \
#   --gpus all \
#   --network origami-contract-test \
#   --shm-size 8g \
#   -v "$(pwd)/vitac_policy_server.py:/app/vitac_policy_server.py:z" \
#   -v "$(pwd)/vitacformer:/app/vitacformer:z" \
#   "$IMAGE"ON_ID="$SESSION" \
#   "$IMAGE"