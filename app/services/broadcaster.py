import asyncio
import logging
from typing import Set

logger = logging.getLogger(__name__)

class EventBroadcaster:
    """Manages active Server-Sent Events (SSE) connections and broadcasts real-time alerts."""
    
    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()

    def connect(self) -> asyncio.Queue:
        """Register a new client connection queue."""
        q = asyncio.Queue(maxsize=50)
        self._subscribers.add(q)
        return q

    def disconnect(self, q: asyncio.Queue):
        """Remove a disconnected client queue."""
        self._subscribers.discard(q)

    @property
    def connection_count(self) -> int:
        return len(self._subscribers)

    async def broadcast(self, event: str, html_content: str):
        """Broadcasts an SSE message formatted with event and data blocks to all active clients."""
        # Clean HTML content so each newline is preceded by 'data: ' according to SSE spec
        lines = [line.strip() for line in html_content.strip().splitlines() if line.strip()]
        data_block = "\n".join(f"data: {line}" for line in lines)
        message = f"event: {event}\n{data_block}\n\n"
        
        dead_queues = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue is full, dropping subscriber")
                dead_queues.append(q)
            except Exception as e:
                logger.error(f"Error broadcasting message to subscriber: {e}")
                dead_queues.append(q)
                
        for q in dead_queues:
            self._subscribers.discard(q)

broadcaster = EventBroadcaster()
