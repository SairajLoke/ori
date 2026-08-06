# docker run -d --name origami-contract-router \
#   --network origami-contract-test \
#   -p 127.0.0.1:17447:7447 \
#   "$ROUTER_IMAGE" \
#   -l tcp/0.0.0.0:7447 \
#   --no-multicast-scouting \
#   --cfg 'transport/shared_memory/enabled:false'

docker stop origami-contract-policy-cpu 

docker rm origami-contract-policy-cpu 

docker build -t "$IMAGE" .

docker run -d --name origami-contract-policy-cpu  \
  --network origami-contract-test \
  --gpus all \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges=true \
  --user 65532:65532 \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=4g \
  --tmpfs /run:rw,noexec,nosuid,nodev,size=64m \
  --shm-size 8g \
  --memory 16g \
  --cpus 4 \
  --pids-limit 512 \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-contract-router:7447 \
  -e ORIGAMI_SESSION_ID="$SESSION" \
  "$IMAGE"

# docker restart origami-contract-policy-cpu
#    

# export TORCHINDUCTOR_CACHE_DIR=/home/sai/.cache/torchinductor



# no constrained run 
# docker run -d --name origami-contract-policy-cpu \
#   --gpus all \
#   --network origami-contract-test \
#   --shm-size 8g \
#   -v "$(pwd)/vitac_policy_server.py:/app/vitac_policy_server.py:z" \
#   -v "$(pwd)/vitacformer:/app/vitacformer:z" \
#   "$IMAGE"ON_ID="$SESSION" \
#   "$IMAGE"


