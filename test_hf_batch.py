#!/usr/bin/env python3
"""
Simple test script to verify HF batch server is working correctly.
"""

import asyncio
from shared import (
    MessageChannel,
    PrefillBatchRequest,
    PrefillBatchResponse,
    VerifyBatchRequest,
    VerifyBatchResponse,
    VerifyItem,
)

async def test_batch():
    # Connect to server
    reader, writer = await asyncio.open_connection("localhost", 7070)
    channel = MessageChannel(reader, writer)

    # Test batch prefill
    print("Testing batch prefill...")
    items = [
        {"stream_id": "s1", "prompt": [100, 200, 300, 400]},
        {"stream_id": "s2", "prompt": [500, 600, 700, 800]},
    ]
    await channel.send(PrefillBatchRequest(items=items))
    resp = await channel.recv()
    print(f"Prefill response: {resp}")

    # Test batch verify
    print("\nTesting batch verify...")
    verify_items = [
        VerifyItem(
            stream_id="s1",
            draft_toks=[101, 102],
            draft_topk_idx=[],
            draft_topk_vals=[],
        ),
        VerifyItem(
            stream_id="s2",
            draft_toks=[501, 502],
            draft_topk_idx=[],
            draft_topk_vals=[],
        ),
    ]
    await channel.send(VerifyBatchRequest(items=verify_items))
    resp = await channel.recv()
    print(f"Verify response: {resp}")

    await channel.close()
    print("\nTest completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_batch())