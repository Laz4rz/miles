from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


def test_old_receivers_without_dtype_metadata_are_rejected(mooncake_module):
    """A receiver's byte counts alone cannot distinguish FP16 from BF16."""

    async def get_info(rank):
        return {"remote_instance_transfer_engine_info": ["receiver", {"w": [1, 4, 2]}]}

    api = SimpleNamespace(get_remote_instance_transfer_engine_info=get_info)
    with pytest.raises(RuntimeError, match="upgrade the inference workers"):
        mooncake_module._query_remote_shard(api, 0, 0)


def test_a_weight_of_another_size_on_the_shard_is_never_written(p2p_sender: Any, mooncake_module: ModuleType) -> None:
    """Writing a weight whose published size differs would run past the target's memory."""
    transport = mooncake_module.MooncakeTransport()
    remote_shard = mooncake_module.RemoteShard(
        rollout_engine_ind=0,
        rollout_engine_rank=0,
        session_id="cell-a-r0",
        weight_locations_by_name={"w": mooncake_module.RemoteWeightLocation(address=0x1000, numel=2, element_size=4)},
        weight_dtypes={"w": "float32"},
    )

    with pytest.raises(AssertionError, match="w is 16 bytes here but 8 bytes on rollout engine 0 rank 0"):
        transport.write([remote_shard], {"w": torch.zeros(4)})
    assert p2p_sender.transfer_engine.writes == []
