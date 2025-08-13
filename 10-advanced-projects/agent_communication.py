#!/usr/bin/env python3
"""
Agent Communication Framework

This module provides advanced communication patterns and protocols
for multi-agent systems using DSPy.

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import dspy

logger = logging.getLogger(__name__)


class CommunicationProtocol(Enum):
    """Communication protocols for agent interaction"""

    DIRECT = "direct"
    BROADCAST = "broadcast"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"
    CONSENSUS = "consensus"


class MessagePriority(Enum):
    """Message priority levels"""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class CommunicationMetrics:
    """Communication performance metrics"""

    messages_sent: int = 0
    messages_received: int = 0
    average_response_time: float = 0.0
    failed_deliveries: int = 0
    bandwidth_usage: int = 0
    last_activity: Optional[datetime] = None


class MessageRouter:
    """Advanced message routing system"""

    def __init__(self):
        self.routes: Dict[str, List[str]] = {}  # topic -> agent_ids
        self.filters: Dict[str, Callable] = {}  # agent_id -> filter_function
        self.metrics: Dict[str, CommunicationMetrics] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def add_route(self, topic: str, agent_id: str):
        """Add a routing rule"""
        if topic not in self.routes:
            self.routes[topic] = []
        if agent_id not in self.routes[topic]:
            self.routes[topic].append(agent_id)

    def remove_route(self, topic: str, agent_id: str):
        """Remove a routing rule"""
        if topic in self.routes and agent_id in self.routes[topic]:
            self.routes[topic].remove(agent_id)

    def set_filter(self, agent_id: str, filter_func: Callable):
        """Set message filter for an agent"""
        self.filters[agent_id] = filter_func

    def route_message(self, topic: str, message: Dict[str, Any]) -> List[str]:
        """Route message to appropriate agents"""
        recipients = self.routes.get(topic, [])

        # Apply filters
        filtered_recipients = []
        for agent_id in recipients:
            if agent_id in self.filters:
                if self.filters[agent_id](message):
                    filtered_recipients.append(agent_id)
            else:
                filtered_recipients.append(agent_id)

        # Update metrics
        for agent_id in filtered_recipients:
            if agent_id not in self.metrics:
                self.metrics[agent_id] = CommunicationMetrics()
            self.metrics[agent_id].messages_sent += 1
            self.metrics[agent_id].last_activity = datetime.utcnow()

        # Store in history
        self.message_history.append(
            {
                "topic": topic,
                "recipients": filtered_recipients,
                "timestamp": datetime.utcnow().isoformat(),
                "message_id": message.get("id", "unknown"),
            }
        )

        # Maintain history size
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history :]

        return filtered_recipients


class ConsensusManager:
    """Manage consensus protocols among agents"""

    def __init__(self):
        self.active_consensus: Dict[str, Dict[str, Any]] = {}
        self.consensus_history: List[Dict[str, Any]] = []

    async def start_consensus(
        self, consensus_id: str, topic: str, participants: List[str], timeout: int = 30
    ) -> str:
        """Start a consensus process"""
        self.active_consensus[consensus_id] = {
            "topic": topic,
            "participants": participants,
            "votes": {},
            "started_at": datetime.utcnow(),
            "timeout": timeout,
            "status": "active",
        }

        logger.info(f"Started consensus {consensus_id} on topic: {topic}")
        return consensus_id

    async def submit_vote(self, consensus_id: str, agent_id: str, vote: Any) -> bool:
        """Submit a vote for consensus"""
        if consensus_id not in self.active_consensus:
            return False

        consensus = self.active_consensus[consensus_id]
        if agent_id not in consensus["participants"]:
            return False

        consensus["votes"][agent_id] = {"vote": vote, "timestamp": datetime.utcnow()}

        # Check if consensus reached
        if len(consensus["votes"]) == len(consensus["participants"]):
            result = await self._evaluate_consensus(consensus_id)
            consensus["result"] = result
            consensus["status"] = "completed"

            # Move to history
            self.consensus_history.append(consensus.copy())
            del self.active_consensus[consensus_id]

            logger.info(f"Consensus {consensus_id} completed with result: {result}")

        return True

    async def _evaluate_consensus(self, consensus_id: str) -> Dict[str, Any]:
        """Evaluate consensus result"""
        consensus = self.active_consensus[consensus_id]
        votes = [vote_data["vote"] for vote_data in consensus["votes"].values()]

        # Simple majority consensus
        vote_counts = {}
        for vote in votes:
            vote_str = str(vote)
            vote_counts[vote_str] = vote_counts.get(vote_str, 0) + 1

        if vote_counts:
            winner = max(vote_counts, key=vote_counts.get)
            return {
                "decision": winner,
                "vote_counts": vote_counts,
                "unanimous": len(set(str(v) for v in votes)) == 1,
            }

        return {"decision": None, "vote_counts": {}, "unanimous": False}


class CommunicationProtocolHandler:
    """Handle different communication protocols"""

    def __init__(
        self, message_router: MessageRouter, consensus_manager: ConsensusManager
    ):
        self.router = message_router
        self.consensus = consensus_manager
        self.protocol_handlers = {
            CommunicationProtocol.DIRECT: self._handle_direct,
            CommunicationProtocol.BROADCAST: self._handle_broadcast,
            CommunicationProtocol.PUBLISH_SUBSCRIBE: self._handle_pubsub,
            CommunicationProtocol.REQUEST_RESPONSE: self._handle_request_response,
            CommunicationProtocol.CONSENSUS: self._handle_consensus,
        }

    async def handle_message(
        self, protocol: CommunicationProtocol, message: Dict[str, Any]
    ) -> List[str]:
        """Handle message according to protocol"""
        if protocol in self.protocol_handlers:
            return await self.protocol_handlers[protocol](message)
        else:
            logger.warning(f"Unknown protocol: {protocol}")
            return []

    async def _handle_direct(self, message: Dict[str, Any]) -> List[str]:
        """Handle direct message protocol"""
        recipient = message.get("recipient")
        return [recipient] if recipient else []

    async def _handle_broadcast(self, message: Dict[str, Any]) -> List[str]:
        """Handle broadcast protocol"""
        # Return all registered agents for the topic
        topic = message.get("topic", "general")
        return self.router.routes.get(topic, [])

    async def _handle_pubsub(self, message: Dict[str, Any]) -> List[str]:
        """Handle publish-subscribe protocol"""
        topic = message.get("topic")
        if topic:
            return self.router.route_message(topic, message)
        return []

    async def _handle_request_response(self, message: Dict[str, Any]) -> List[str]:
        """Handle request-response protocol"""
        # Similar to direct, but with response tracking
        recipient = message.get("recipient")
        if recipient and message.get("requires_response"):
            # Track pending response
            pass
        return [recipient] if recipient else []

    async def _handle_consensus(self, message: Dict[str, Any]) -> List[str]:
        """Handle consensus protocol"""
        consensus_id = message.get("consensus_id")
        if consensus_id and "vote" in message:
            agent_id = message.get("sender_id")
            vote = message.get("vote")
            await self.consensus.submit_vote(consensus_id, agent_id, vote)

        # Return participants for consensus updates
        if consensus_id in self.consensus.active_consensus:
            return self.consensus.active_consensus[consensus_id]["participants"]
        return []


# DSPy Signatures for Communication
class MessageTranslation(dspy.Signature):
    """Translate messages between different agent communication styles"""

    source_message: str = dspy.InputField(desc="Original message content")
    source_style: str = dspy.InputField(desc="Source communication style")
    target_style: str = dspy.InputField(desc="Target communication style")
    translated_message: str = dspy.OutputField(desc="Translated message")


class ConflictResolution(dspy.Signature):
    """Resolve conflicts in agent communication"""

    conflict_description: str = dspy.InputField(desc="Description of the conflict")
    agent_positions: str = dspy.InputField(desc="Different agent positions")
    context: str = dspy.InputField(desc="Situational context")
    resolution: str = dspy.OutputField(desc="Proposed conflict resolution")
    compromise: str = dspy.OutputField(desc="Compromise solution")


class CommunicationOptimization(dspy.Signature):
    """Optimize communication patterns for efficiency"""

    current_patterns: str = dspy.InputField(desc="Current communication patterns")
    performance_metrics: str = dspy.InputField(desc="Performance metrics")
    constraints: str = dspy.InputField(desc="System constraints")
    optimized_patterns: str = dspy.OutputField(desc="Optimized communication patterns")
    expected_improvements: str = dspy.OutputField(
        desc="Expected performance improvements"
    )


class AdvancedCommunicationManager:
    """Advanced communication management for multi-agent systems"""

    def __init__(self):
        self.router = MessageRouter()
        self.consensus = ConsensusManager()
        self.protocol_handler = CommunicationProtocolHandler(
            self.router, self.consensus
        )
        self.active_conversations: Dict[str, Dict[str, Any]] = {}

        # Initialize DSPy modules
        self.message_translator = dspy.ChainOfThought(MessageTranslation)
        self.conflict_resolver = dspy.ChainOfThought(ConflictResolution)
        self.communication_optimizer = dspy.ChainOfThought(CommunicationOptimization)

    async def setup_communication_topology(self, topology: Dict[str, List[str]]):
        """Set up communication topology"""
        for topic, agents in topology.items():
            for agent in agents:
                self.router.add_route(topic, agent)

        logger.info(f"Set up communication topology with {len(topology)} topics")

    async def start_conversation(
        self, conversation_id: str, participants: List[str], topic: str
    ) -> str:
        """Start a structured conversation"""
        self.active_conversations[conversation_id] = {
            "participants": participants,
            "topic": topic,
            "messages": [],
            "started_at": datetime.utcnow(),
            "status": "active",
        }

        # Set up routing for this conversation
        self.router.add_route(f"conversation_{conversation_id}", participants[0])
        for participant in participants:
            self.router.add_route(f"conversation_{conversation_id}", participant)

        return conversation_id

    async def translate_message(
        self, message: str, source_style: str, target_style: str
    ) -> str:
        """Translate message between communication styles"""
        result = self.message_translator(
            source_message=message, source_style=source_style, target_style=target_style
        )
        return result.translated_message

    async def resolve_conflict(
        self, conflict_desc: str, agent_positions: List[str], context: str
    ) -> Dict[str, str]:
        """Resolve communication conflicts"""
        positions_str = "; ".join(agent_positions)

        result = self.conflict_resolver(
            conflict_description=conflict_desc,
            agent_positions=positions_str,
            context=context,
        )

        return {"resolution": result.resolution, "compromise": result.compromise}

    async def optimize_communication(
        self, current_patterns: Dict[str, Any], metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize communication patterns"""
        patterns_str = json.dumps(current_patterns)
        metrics_str = json.dumps(metrics)

        result = self.communication_optimizer(
            current_patterns=patterns_str,
            performance_metrics=metrics_str,
            constraints="Low latency, high reliability",
        )

        try:
            optimized = json.loads(result.optimized_patterns)
            return {"patterns": optimized, "improvements": result.expected_improvements}
        except json.JSONDecodeError:
            return {
                "patterns": current_patterns,
                "improvements": result.expected_improvements,
            }

    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get communication performance metrics"""
        total_messages = sum(m.messages_sent for m in self.router.metrics.values())
        active_agents = len(
            [m for m in self.router.metrics.values() if m.last_activity]
        )

        return {
            "total_messages": total_messages,
            "active_agents": active_agents,
            "active_conversations": len(self.active_conversations),
            "active_consensus": len(self.consensus.active_consensus),
            "routing_rules": len(self.router.routes),
            "message_history_size": len(self.router.message_history),
        }


async def demonstrate_communication_framework():
    """Demonstrate the communication framework"""
    print("=== Agent Communication Framework Demo ===")

    # Create communication manager
    comm_manager = AdvancedCommunicationManager()

    # Set up communication topology
    topology = {
        "research": ["researcher_001", "researcher_002", "analyst_001"],
        "coordination": ["coordinator_001", "synthesizer_001"],
        "evaluation": ["critic_001", "analyst_001"],
    }

    await comm_manager.setup_communication_topology(topology)
    print(f"Set up communication topology: {list(topology.keys())}")

    # Start a conversation
    conversation_id = await comm_manager.start_conversation(
        "conv_001",
        ["researcher_001", "analyst_001", "synthesizer_001"],
        "AI in Healthcare Research",
    )
    print(f"Started conversation: {conversation_id}")

    # Demonstrate message translation
    original_message = "The research findings indicate significant potential."
    translated = await comm_manager.translate_message(
        original_message, "formal_academic", "casual_collaborative"
    )
    print(f"Message translation:")
    print(f"  Original: {original_message}")
    print(f"  Translated: {translated}")

    # Demonstrate conflict resolution
    conflict = "Disagreement on research methodology approach"
    positions = [
        "Agent A prefers quantitative analysis",
        "Agent B prefers qualitative analysis",
        "Agent C suggests mixed methods",
    ]

    resolution = await comm_manager.resolve_conflict(
        conflict, positions, "Healthcare AI research project"
    )
    print(f"\nConflict Resolution:")
    print(f"  Resolution: {resolution['resolution']}")
    print(f"  Compromise: {resolution['compromise']}")

    # Start consensus process
    consensus_id = await comm_manager.consensus.start_consensus(
        "consensus_001",
        "Research Priority Selection",
        ["researcher_001", "analyst_001", "critic_001"],
    )

    # Submit votes
    await comm_manager.consensus.submit_vote(consensus_id, "researcher_001", "Option A")
    await comm_manager.consensus.submit_vote(consensus_id, "analyst_001", "Option A")
    await comm_manager.consensus.submit_vote(consensus_id, "critic_001", "Option B")

    print(f"\nConsensus completed for: {consensus_id}")

    # Get metrics
    metrics = comm_manager.get_communication_metrics()
    print(f"\nCommunication Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\nCommunication framework demo completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_communication_framework())
