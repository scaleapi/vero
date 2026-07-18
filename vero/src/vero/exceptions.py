import asyncio


class StreamEventTimeout(asyncio.TimeoutError):
    pass
