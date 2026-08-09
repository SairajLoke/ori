filepath="/home/sai/Desktop/ORI/ori/origami-inference-kit-participant/sharpa_north_ces_lite_sdk-main/examples/check_zenoh_policy.py"
srcpath="/home/sai/Desktop/ORI/ori/submission-0/scripts/run.sh"

source $srcpath

uv run --no-sync python $filepath \
  --endpoint tcp/127.0.0.1:17447 \
  --session-id "$SESSION" \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 25