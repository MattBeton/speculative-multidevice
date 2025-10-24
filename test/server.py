import asyncio

from shared import MessageChannel, PrefillRequest, ResetRequest, VerifyRequest, VerifyResponse


class VerifierSession:
    """Minimal session state for handling verifier requests."""

    def __init__(self) -> None:
        self._prompt: list[int] = []
        self._accepted_tokens = 0

    def reset(self) -> None:
        self._prompt = []
        self._accepted_tokens = 0

    def prefill(self, request: PrefillRequest) -> None:
        self._prompt = request.prompt

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        # TODO: replace with real verification logic.
        accepted_len = len(request.draft_toks)
        self._accepted_tokens += accepted_len
        return VerifyResponse(accepted_len=accepted_len)


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    print(f"client connected: {peer}")
    channel = MessageChannel(reader, writer)
    session = VerifierSession()
    try:
        while True:
            message = await channel.recv()
            if message is None:
                print(f"client disconnected: {peer}")
                break

            if isinstance(message, ResetRequest):
                session.reset()
                print("reset received")
            elif isinstance(message, PrefillRequest):
                session.prefill(message)
                print(f"prefill received ({len(message.prompt)} tokens)")
            elif isinstance(message, VerifyRequest):
                response = session.verify(message)
                await channel.send(response)
                print(f"verify received -> accepted {response.accepted_len}")
            else:
                raise RuntimeError(f"unhandled message type: {type(message)!r}")
    finally:
        await channel.close()


async def main() -> None:
    server = await asyncio.start_server(handle_client, "127.0.0.1", 7070)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Verifier listening on {addrs}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
