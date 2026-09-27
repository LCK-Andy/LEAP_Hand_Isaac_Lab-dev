"""Pure configuration and resume boundaries for the standalone TD-MPC2 runner."""

import json


def canonical_contract(contract):
    return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))


def recipient_trainer_state(payload):
    metadata = payload["pbt"]
    if "recipient_frame" in metadata:
        if "recipient_trainer_state" not in metadata:
            raise ValueError("Staged restart lacks recipient trainer accounting")
        state = metadata["recipient_trainer_state"]
    else:
        state = payload["training_state"]["trainer"]
    if state["frame"] != metadata.get("recipient_frame", metadata["frame"]):
        raise ValueError("Checkpoint trainer and recipient frame disagree")
    if state["updates"] != metadata.get("recipient_epoch", metadata["epoch"]):
        raise ValueError("Checkpoint trainer and recipient updates disagree")
    return state
