"""
🔄 Event Loop for Ada
Handles async event processing and coordination
"""

import asyncio
import time
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    """Types of events Ada can handle"""
    USER_INPUT = "user_input"
    NEURAL_RESPONSE = "neural_response"
    DATABASE_SAVE = "database_save"
    TRAINING_UPDATE = "training_update"
    SYSTEM_SHUTDOWN = "system_shutdown"
    HEARTBEAT = "heartbeat"

@dataclass
class AdaEvent:
    """Event data structure for Ada's event system"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float
    source: str = "system"

class EventLoop:
    """Event loop for managing Ada's async operations"""
    
    def __init__(self):
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.event_queue: List[AdaEvent] = []
        self.running = False
        self.processed_events = 0
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default event handlers"""
        self.register_handler(EventType.HEARTBEAT, self._handle_heartbeat)
        self.register_handler(EventType.SYSTEM_SHUTDOWN, self._handle_shutdown)
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler for a specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def emit_event(self, event: AdaEvent):
        """Emit an event to be processed"""
        self.event_queue.append(event)
        if not self.running:
            self.process_event(event)
    
    def emit(self, event_type: EventType, data: Dict[str, Any], source: str = "system"):
        """Convenience method to emit an event"""
        event = AdaEvent(
            event_type=event_type,
            data=data,
            timestamp=time.time(),
            source=source
        )
        self.emit_event(event)
    
    def process_event(self, event: AdaEvent):
        """Process a single event"""
        try:
            handlers = self.event_handlers.get(event.event_type, [])
            for handler in handlers:
                handler(event)
            self.processed_events += 1
        except Exception as e:
            print(f"❌ Error processing event {event.event_type}: {e}")
    
    def process_queue(self):
        """Process all events in the queue"""
        while self.event_queue:
            event = self.event_queue.pop(0)
            self.process_event(event)
    
    def start(self):
        """Start the event loop"""
        self.running = True
        print("🔄 Ada event loop started")
        
        # Emit startup heartbeat
        self.emit(EventType.HEARTBEAT, {"status": "started"})
    
    def stop(self):
        """Stop the event loop"""
        self.running = False
        print("🔄 Ada event loop stopped")
        
        # Emit shutdown event
        self.emit(EventType.SYSTEM_SHUTDOWN, {"events_processed": self.processed_events})
    
    def _handle_heartbeat(self, event: AdaEvent):
        """Handle heartbeat events"""
        status = event.data.get("status", "unknown")
        print(f"💓 Heartbeat: {status}")
    
    def _handle_shutdown(self, event: AdaEvent):
        """Handle system shutdown"""
        events_processed = event.data.get("events_processed", 0)
        print(f"🛑 Shutdown: Processed {events_processed} events")
    
    def get_statistics(self) -> Dict:
        """Get event loop statistics"""
        return {
            "running": self.running,
            "events_processed": self.processed_events,
            "queue_size": len(self.event_queue),
            "registered_handlers": sum(len(handlers) for handlers in self.event_handlers.values())
        }

class AdaEventManager:
    """Manager for coordinating Ada's events with other systems"""
    
    def __init__(self, event_loop: EventLoop):
        self.event_loop = event_loop
        self.subscribers: List[Callable] = []
    
    def subscribe(self, callback: Callable):
        """Subscribe to all events"""
        self.subscribers.append(callback)
    
    def notify_subscribers(self, event: AdaEvent):
        """Notify all subscribers about an event"""
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception as e:
                print(f"❌ Error in subscriber notification: {e}")
    
    def create_user_input_event(self, user_input: str):
        """Create a user input event"""
        self.event_loop.emit(EventType.USER_INPUT, {
            "input": user_input,
            "length": len(user_input)
        })
    
    def create_response_event(self, response: str, confidence: float = 0.5):
        """Create a neural response event"""
        self.event_loop.emit(EventType.NEURAL_RESPONSE, {
            "response": response,
            "confidence": confidence,
            "length": len(response)
        })
    
    def create_database_event(self, operation: str, success: bool, details: Dict = None):
        """Create a database event"""
        self.event_loop.emit(EventType.DATABASE_SAVE, {
            "operation": operation,
            "success": success,
            "details": details or {}
        })

# Example usage and testing
def test_event_loop():
    """Test the event loop functionality"""
    print("🧪 Testing Ada Event Loop...")
    
    # Create event loop
    loop = EventLoop()
    manager = AdaEventManager(loop)
    
    # Test custom event handler
    def custom_handler(event: AdaEvent):
        print(f"📢 Custom handler received: {event.event_type.value}")
        if event.event_type == EventType.USER_INPUT:
            print(f"   User said: {event.data.get('input', 'Unknown')}")
    
    loop.register_handler(EventType.USER_INPUT, custom_handler)
    
    # Start loop
    loop.start()
    
    # Emit test events
    loop.emit(EventType.USER_INPUT, {"input": "Hello Ada!"})
    manager.create_response_event("Hello! How can I help?", 0.9)
    loop.emit(EventType.DATABASE_SAVE, {"operation": "save_conversation", "success": True})
    
    # Process queue
    loop.process_queue()
    
    # Show statistics
    stats = loop.get_statistics()
    print(f"📊 Event loop stats: {stats}")
    
    # Stop loop
    loop.stop()
    
    print("✅ Event loop test completed!")

if __name__ == "__main__":
    test_event_loop()