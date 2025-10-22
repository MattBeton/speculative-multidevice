import asyncio
import json
import time

async def main():
    seq_id = "seq-1"
    prompt = "You are EXO. Summarize: speculative decoding on two devices."
    verifier_host, verifier_port = "127.0.0.1", 7070

    reader, writer = await asyncio.open_connection(verifier_host, verifier_port)

    # HELLO
    start = time.perf_counter()
    writer.write(b'{"type":"HELLO"}\n')
    await writer.drain()
    response = await reader.readline()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    if not response:
        print("server closed connection without response")
    else:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            print(f"server sent invalid JSON: {response!r}")
        else:
            print(f"server response: {payload} (round-trip {elapsed_ms:.2f} ms)")

    writer.close(); await writer.wait_closed()
    # Decode final text just to see something:

if __name__ == "__main__":
    asyncio.run(main())
