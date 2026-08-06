source set_local_contract_vars.sh #to confirm path here


docker network create origami-contract-test

docker run -d --name origami-contract-router \
  --network origami-contract-test \
  -p 127.0.0.1:17447:7447 \
  "$ROUTER_IMAGE" \
  -l tcp/0.0.0.0:7447 \
  --no-multicast-scouting \
  --cfg 'transport/shared_memory/enabled:false'


# then after unning policycontainer do this 
# cd sharpa_north_ces_lite_sdk-main
# uv run --no-sync python examples/check_zenoh_policy.py \
#   --endpoint tcp/127.0.0.1:17447 \
#   --session-id "$SESSION" \
#   --timeout 180 \
#   --requests 3 \
#   --expected-horizon 25
