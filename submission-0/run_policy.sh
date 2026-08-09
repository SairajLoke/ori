# export TORCHINDUCTOR_CACHE_DIR=/app/.cache/torchinductor

docker stop origami-contract-policy 

docker rm origami-contract-policy 

docker build -t vitac-policy:dev .

docker run -d --name origami-contract-policy \
  --network origami-contract-test \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges=true \
  --user 65532:65532 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \
  --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
  --tmpfs /app/.cache:rw,exec,nosuid,nodev,size=2g \
  --shm-size 2g \
  --memory 10g \
  --cpus 4 \
  --pids-limit 512 \
  -v "$(pwd)/vitac_policy_server.py:/app/vitac_policy_server.py:ro" \
  -v "$(pwd)/vitacformer:/app/vitacformer:ro" \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-contract-router:7447 \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  "$IMAGE"

# docker restart origami-contract-policy
#    --gpus all \

# export TORCHINDUCTOR_CACHE_DIR=/home/sai/.cache/torchinductor