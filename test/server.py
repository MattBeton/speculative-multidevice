# verifier_server.py
import asyncio, json
from typing import Dict, Tuple, List
import numpy as np
from mlx_lm import load
import mlx.core as mx



# ------------ Networking (asyncio TCP for simplicity) ------------
async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    peer = writer.get_extra_info("peername")
    print(f"client connected: {peer}")
    # simple line-delimited JSON protocol: {"type": "...", "payload": {...}}
    while True:
        line = await reader.readline()
        if not line:
            print(f"client disconnected: {peer}")
            break
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(f"malformed message from {peer}: {line!r}")
            writer.write(b'{"type":"ERROR","payload":{"reason":"invalid json"}}\n')
            await writer.drain()
            continue

        if msg["type"] == "HELLO":
            writer.write(b'{"type":"OK"}\n'); await writer.drain()
            print(f"HELLO received from {peer}")
        else:
            print(f"unknown message type from {peer}: {msg!r}")
            writer.write(b'{"type":"ERROR","payload":{"reason":"unknown message type"}}\n')
            await writer.drain()

    writer.close(); await writer.wait_closed()

async def main():
    server = await asyncio.start_server(lambda r,w: handle_client(r,w), "127.0.0.1", 7070)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"Verifier listening on {addrs}")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
