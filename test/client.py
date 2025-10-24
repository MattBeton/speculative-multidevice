import asyncio

from shared import MessageChannel, PrefillRequest, ResetRequest, VerifyRequest, VerifyResponse


class VerifierClient:
    """Tiny helper encapsulating the client-side protocol."""

    def __init__(self, channel: MessageChannel):
        self._channel = channel

    async def reset(self) -> None:
        await self._channel.send(ResetRequest())

    async def prefill(self, prompt: list[int]) -> None:
        await self._channel.send(PrefillRequest(prompt=prompt))

    async def verify(
        self,
        draft_toks: list[int],
        draft_topk_vals: list[list[float]],
        draft_topk_idx: list[list[int]],
    ) -> int:
        await self._channel.send(
            VerifyRequest(
                draft_toks=draft_toks,
                draft_topk_vals=draft_topk_vals,
                draft_topk_idx=draft_topk_idx,
            )
        )
        response = await self._channel.recv()
        if not isinstance(response, VerifyResponse):
            raise RuntimeError(f"expected VerifyResponse, got {type(response)!r}")
        return response.accepted_len


async def main() -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", 7070)
    channel = MessageChannel(reader, writer)
    client = VerifierClient(channel)

    await client.reset()
    await client.prefill(prompt=[1, 2, 3, 4])

    accepted = await client.verify(
        draft_toks=[5, 6],
        draft_topk_vals=[[0.9, 0.05, 0.05], [0.85, 0.10, 0.05]],
        draft_topk_idx=[[5, 7, 8], [6, 3, 9]],
    )
    print(f"verify accepted tokens: {accepted}")

    await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
