#!/usr/bin/env python3
"""
ViTacFormer Policy Server for Origami Competition
Implements origami-zenoh-v1 protocol for robot inference

This server:
1. Connects to organizer's Zenoh router
2. Declares queryables: metadata, reset, infer
3. Runs ViTacFormer model inference with history buffers
4. Returns actions in competition format (T, 65) float32 radians

Environment variables:
    ORIGAMI_ZENOH_ENDPOINT: tcp://<host>:<port>
    ORIGAMI_SESSION_ID: Opaque session identifier
"""

import os
import sys
import json
import time
import logging
import signal
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np

# Zenoh imports
try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not installed. Run: pip install zenoh")
    sys.exit(1)

# MessagePack imports
try:
    import msgpack
    import msgpack_numpy
except ImportError:
    print("ERROR: msgpack-numpy not installed. Run: pip install msgpack msgpack-numpy")
    sys.exit(1)

# ViTacFormer imports
from vitac_policy import ViTacFormerAdapter, load_vitacformer_model, HistoryBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('vitacformer_server')

# ============================================================================
# Competition Constants
# ============================================================================

# Protocol version for Zenoh transport
ZENOH_PROTOCOL_VERSION = "origami-zenoh-v1"

# Semantic protocol version for policy metadata
POLICY_PROTOCOL_VERSION = "origami-v1"

# Action specification
ACTION_DIM = 65
ACTION_HORIZON = 25  # Must match what model can produce
ACTION_TYPE = "absolute_joint_position"
ACTION_UNITS = "radians"

# Joint names EXACTLY as specified in robot_io_spec.md
# Order: left_arm(7) + left_hand(22) + right_arm(7) + right_hand(22) + motor(7)
JOINT_NAMES = [
    # Left arm (0:7)
    "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
    "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6",
    "left_arm_joint_7",
    # Left hand (7:29)
    "left_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
    # Right arm (29:36)
    "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
    "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6",
    "right_arm_joint_7",
    # Right hand (36:58)
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE",
    "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA",
    "right_pinky_PIP", "right_pinky_DIP",
    # Motor/neck (58:65)
    "lower_body_joint_1", "lower_body_joint_2", "lower_body_joint_3",
    "lower_body_joint_4", "lower_body_joint_5",
    "neck_joint_1", "neck_joint_2",
]

assert len(JOINT_NAMES) == ACTION_DIM, f"Expected {ACTION_DIM} joint names, got {len(JOINT_NAMES)}"

# ============================================================================
# MessagePack Codec
# ============================================================================

def encode_array(arr: np.ndarray) -> Dict[bytes, Any]:
    """Encode numpy array to MessagePack with safe numpy codec."""
    return {
        b"__ndarray__": True,
        b"data": arr.tobytes(order="C"),
        b"dtype": arr.dtype.str,
        b"shape": arr.shape,
    }


def decode_array(obj: Dict[bytes, Any]) -> np.ndarray:
    """Decode numpy array from MessagePack."""
    if not isinstance(obj, dict) or not obj.get(b"__ndarray__"):
        raise ValueError(f"Expected ndarray object, got {type(obj)}")
    
    return np.frombuffer(obj[b"data"], dtype=obj[b"dtype"]).reshape(obj[b"shape"])


# Custom MessagePack extension handler for numpy
def default_ext_handler(code: int, data: bytes) -> Any:
    """Handle MessagePack extensions."""
    if code == 0x41:  # numpy extension
        return msgpack_numpy.unpackb(data)
    return msgpack.ExtType(code, data)


# ============================================================================
# Policy Server Class
# ============================================================================

class ViTacFormerPolicyServer:
    """
    Zenoh policy server for ViTacFormer model.
    
    Declares three queryables:
    - origami-zenoh-v1/metadata: Return policy metadata
    - origami-zenoh-v1/reset: Reset episode state
    - origami-zenoh-v1/infer: Run inference on observation
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        use_tactile: bool = True,
        action_horizon: int = ACTION_HORIZON,
        device: Optional[str] = None
    ):
        """
        Initialize the policy server.
        
        Args:
            checkpoint_path: Path to ViTacFormer checkpoint
            use_tactile: Whether to use tactile observations
            action_horizon: Number of action steps to predict
            device: Target device ('cuda', 'cpu', or None for auto)
        """
        self.checkpoint_path = checkpoint_path
        self.use_tactile = use_tactile
        self.action_horizon = action_horizon
        self.device = device
        
        self.adapter: Optional[ViTacFormerAdapter] = None
        self.zenoh_session: Optional[zenoh.Session] = None
        self.running = False
        self.request_count = 0
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self) -> None:
        """Load ViTacFormer model and create adapter."""
        logger.info(f"Loading ViTacFormer model from: {self.checkpoint_path}")
        
        model = load_vitacformer_model(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            use_tactile=self.use_tactile
        )
        
        self.adapter = ViTacFormerAdapter(
            model=model,
            use_tactile=self.use_tactile,
            action_horizon=self.action_horizon,
            device=self.device
        )
        
        logger.info("Model loaded successfully")
    
    def _connect_zenoh(self, endpoint: str) -> None:
        """
        Connect to Zenoh router.
        
        Args:
            endpoint: Zenoh endpoint (e.g., "tcp/192.168.1.100:7447")
        """
        logger.info(f"Connecting to Zenoh endpoint: {endpoint}")
        
        # Configure Zenoh session
        config = zenoh.Config()
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        
        # Open session
        self.zenoh_session = zenoh.open(config)
        logger.info("Zenoh session opened")
        
        # Declare queryables
        self._declare_queryables()
    
    def _declare_queryables(self) -> None:
        """Declare the three required queryables."""
        # Metadata queryable
        self.zenoh_session.declare_queryable(
            f"{ZENOH_PROTOCOL_VERSION}/metadata",
            self._on_metadata_query
        )
        logger.info(f"Declared queryable: {ZENOH_PROTOCOL_VERSION}/metadata")
        
        # Reset queryable
        self.zenoh_session.declare_queryable(
            f"{ZENOH_PROTOCOL_VERSION}/reset",
            self._on_reset_query
        )
        logger.info(f"Declared queryable: {ZENOH_PROTOCOL_VERSION}/reset")
        
        # Infer queryable
        self.zenoh_session.declare_queryable(
            f"{ZENOH_PROTOCOL_VERSION}/infer",
            self._on_infer_query
        )
        logger.info(f"Declared queryable: {ZENOH_PROTOCOL_VERSION}/infer")
    
    def _on_metadata_query(self, query: zenoh.Query) -> None:
        """Handle metadata query."""
        try:
            # Parse request
            request = msgpack.loads(query.payload, ext_hook=default_ext_handler)
            
            # Validate envelope
            self._validate_envelope(request, "metadata")
            
            # Build metadata response
            response = {
                "protocol_version": ZENOH_PROTOCOL_VERSION,
                "operation": "metadata",
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "metadata": {
                    "protocol_version": POLICY_PROTOCOL_VERSION,
                    "action_dim": ACTION_DIM,
                    "action_horizon": self.action_horizon,
                    "action_type": ACTION_TYPE,
                    "action_units": ACTION_UNITS,
                    "joint_names": JOINT_NAMES,
                },
            }
            
            # Encode and send response
            payload = msgpack.dumps(response, use_bin_type=True)
            query.reply(
                f"{ZENOH_PROTOCOL_VERSION}/metadata",
                payload
            )
            
            logger.info(f"Metadata response sent: horizon={self.action_horizon}")
            
        except Exception as e:
            logger.error(f"Metadata query failed: {e}")
            self._send_error(query, "metadata", request if 'request' in dir() else {}, "INTERNAL", str(e))
    
    def _on_reset_query(self, query: zenoh.Query) -> None:
        """Handle reset query."""
        try:
            # Parse request
            request = msgpack.loads(query.payload, ext_hook=default_ext_handler)
            
            # Validate envelope
            self._validate_envelope(request, "reset")
            
            # Reset adapter (clears history buffers)
            if self.adapter:
                self.adapter.reset()
            
            # Build response
            response = {
                "protocol_version": ZENOH_PROTOCOL_VERSION,
                "operation": "reset",
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "ok": True,
            }
            
            # Send response
            payload = msgpack.dumps(response, use_bin_type=True)
            query.reply(
                f"{ZENOH_PROTOCOL_VERSION}/reset",
                payload
            )
            
            logger.info("Reset acknowledged")
            
        except Exception as e:
            logger.error(f"Reset query failed: {e}")
            self._send_error(query, "reset", request if 'request' in dir() else {}, "INTERNAL", str(e))
    
    def _on_infer_query(self, query: zenoh.Query) -> None:
        """Handle inference query."""
        start_time = time.monotonic()
        
        try:
            # Parse request
            request = msgpack.loads(query.payload, ext_hook=default_ext_handler)
            
            # Validate envelope
            self._validate_envelope(request, "infer")
            
            # Validate observation
            obs = request.get("observation")
            if not obs:
                raise ValueError("Missing 'observation' in request")
            
            # Run inference
            self.request_count += 1
            actions = self.adapter.infer(obs)
            
            # Build response
            infer_time_ms = (time.monotonic() - start_time) * 1000
            
            response = {
                "protocol_version": ZENOH_PROTOCOL_VERSION,
                "operation": "infer",
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "actions": actions,
                "policy_timing": {
                    "infer_ms": infer_time_ms,
                },
            }
            
            # Encode with numpy support
            # Custom encoder for numpy arrays
            def numpy_encoder(obj):
                if isinstance(obj, np.ndarray):
                    return encode_array(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            payload = msgpack.dumps(response, default=numpy_encoder, use_bin_type=True)
            
            query.reply(
                f"{ZENOH_PROTOCOL_VERSION}/infer",
                payload
            )
            
            logger.info(
                f"Inference request={self.request_count} "
                f"action_shape={actions.shape} "
                f"model_ms={infer_time_ms:.1f}"
            )
            
        except Exception as e:
            logger.error(f"Inference query failed: {e}")
            traceback.print_exc()
            self._send_error(query, "infer", request if 'request' in dir() else {}, "INFERENCE_FAILED", str(e))
    
    def _validate_envelope(self, request: Dict, operation: str) -> None:
        """Validate request envelope fields."""
        required_fields = ["protocol_version", "operation", "request_id", "session_id"]
        
        for field in required_fields:
            if field not in request:
                raise ValueError(f"Missing required field: {field}")
        
        if request["protocol_version"] != ZENOH_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported protocol version: {request['protocol_version']}, "
                f"expected {ZENOH_PROTOCOL_VERSION}"
            )
        
        if request["operation"] != operation:
            raise ValueError(
                f"Operation mismatch: {request['operation']}, expected {operation}"
            )
    
    def _send_error(
        self,
        query: zenoh.Query,
        operation: str,
        request: Dict,
        code: str,
        message: str,
        retryable: bool = False
    ) -> None:
        """Send structured error response."""
        response = {
            "protocol_version": ZENOH_PROTOCOL_VERSION,
            "operation": operation,
            "request_id": request.get("request_id", "unknown"),
            "session_id": request.get("session_id", "unknown"),
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
        }
        
        payload = msgpack.dumps(response, use_bin_type=True)
        query.reply(f"{ZENOH_PROTOCOL_VERSION}/{operation}", payload)
    
    def run(self, endpoint: str, session_id: str) -> None:
        """
        Run the policy server.
        
        Args:
            endpoint: Zenoh endpoint
            session_id: Session identifier (for validation)
        """
        # Validate session ID
        if not session_id:
            raise ValueError("ORIGAMI_SESSION_ID is required")
        
        logger.info(f"Starting ViTacFormer Policy Server")
        logger.info(f"  Endpoint: {endpoint}")
        logger.info(f"  Session ID: {session_id}")
        logger.info(f"  Checkpoint: {self.checkpoint_path}")
        logger.info(f"  Action horizon: {self.action_horizon}")
        logger.info(f"  Use tactile: {self.use_tactile}")
        
        # Connect to Zenoh
        self._connect_zenoh(endpoint)
        
        # Set up signal handlers
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        logger.info("Server ready, waiting for queries...")
        
        # Main loop (Zenoh handles queries asynchronously)
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt")
        finally:
            self._shutdown()
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down...")
        
        if self.zenoh_session:
            self.zenoh_session.close()
            logger.info("Zenoh session closed")
        
        logger.info("Shutdown complete")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Read environment variables
    endpoint = os.environ.get("ORIGAMI_ZENOH_ENDPOINT")
    session_id = os.environ.get("ORIGAMI_SESSION_ID")
    
    # Validate required environment variables
    if not endpoint:
        logger.error("ERROR: ORIGAMI_ZENOH_ENDPOINT is required")
        logger.error("  Example: ORIGAMI_ZENOH_ENDPOINT=tcp/192.168.1.100:7447")
        sys.exit(1)
    
    if not session_id:
        logger.error("ERROR: ORIGAMI_SESSION_ID is required")
        sys.exit(1)
    
    # Parse endpoint format (tcp/host:port)
    if not endpoint.startswith("tcp/"):
        logger.warning(f"Endpoint format may be incorrect: {endpoint}")
    
    # Get checkpoint path from environment or use default
    checkpoint_path = os.environ.get("VITAC_CKPT_PATH",None)
    
    # Get optional settings
    use_tactile = os.environ.get("VITACFORMER_USE_TACTILE", "true").lower() == "true"
    action_horizon = int(os.environ.get("ORIGAMI_ACTION_HORIZON", str(ACTION_HORIZON)))
    device = os.environ.get("VITACFORMER_DEVICE", None)
    
    # Create and run server
    server = ViTacFormerPolicyServer(
        checkpoint_path=checkpoint_path,
        use_tactile=use_tactile,
        action_horizon=action_horizon,
        device=device
    )
    
    server.run(endpoint=endpoint, session_id=session_id)


if __name__ == "__main__":
    main()
