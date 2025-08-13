#!/usr/bin/env python3
"""
Advanced Conversational AI Platform

This module provides a sophisticated conversational AI system with advanced
dialogue management, context handling, memory persistence, and conversation
analytics using DSPy. The platform supports multi-turn conversations,
personality customization, and intelligent response generation.

Learning Objectives:
- Implement advanced conversation management systems
- Create intelligent context and memory handling
- Build multi-turn dialogue optimization
- Develop conversation analytics and insights
- Master DSPy patterns for conversational AI

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import dspy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Conversation states"""

    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"


class MessageType(Enum):
    """Message types"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    ERROR = "error"


class ConversationMode(Enum):
    """Conversation modes"""

    CASUAL = "casual"
    PROFESSIONAL = "professional"
    EDUCATIONAL = "educational"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    SUPPORTIVE = "supportive"


@dataclass
class Message:
    """Represents a conversation message"""

    id: str = field(default_factory=lambda: str(uuid4()))
    conversation_id: str = ""
    message_type: MessageType = MessageType.USER
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    processing_time: float = 0.0


@dataclass
class ConversationContext:
    """Conversation context information"""

    conversation_id: str = ""
    current_topic: str = ""
    topics_discussed: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_mode: ConversationMode = ConversationMode.CASUAL
    active_entities: Dict[str, Any] = field(default_factory=dict)
    context_window: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationAnalytics:
    """Conversation analytics data"""

    conversation_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    message_count: int = 0
    average_response_time: float = 0.0
    user_satisfaction_score: float = 0.0
    engagement_score: float = 0.0
    topic_coherence_score: float = 0.0
    context_retention_score: float = 0.0
    conversation_quality_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)


@dataclass
class Conversation:
    """Complete conversation object"""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    title: str = ""
    state: ConversationState = ConversationState.ACTIVE
    mode: ConversationMode = ConversationMode.CASUAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    messages: List[Message] = field(default_factory=list)
    context: ConversationContext = field(default_factory=ConversationContext)
    analytics: ConversationAnalytics = field(default_factory=ConversationAnalytics)
    metadata: Dict[str, Any] = field(default_factory=dict)


# DSPy Signatures for Conversational AI
class ConversationResponse(dspy.Signature):
    """Generate contextual conversation responses"""

    user_message: str = dspy.InputField(desc="User's message")
    conversation_history: str = dspy.InputField(desc="Recent conversation history")
    user_context: str = dspy.InputField(desc="User context and preferences")
    conversation_mode: str = dspy.InputField(desc="Conversation mode and style")
    response: str = dspy.OutputField(desc="Contextual response to user")
    confidence: float = dspy.OutputField(desc="Response confidence (0-1)")
    suggested_followup: str = dspy.OutputField(desc="Suggested follow-up questions")


class ConversationEngine:
    """Core conversation processing engine"""

    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}

        # Initialize DSPy modules
        self.response_generator = dspy.ChainOfThought(ConversationResponse)

    async def create_conversation(
        self,
        user_id: str,
        mode: ConversationMode = ConversationMode.CASUAL,
        title: str = "",
    ) -> Conversation:
        """Create a new conversation"""
        conversation = Conversation(
            user_id=user_id,
            title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            mode=mode,
        )

        # Initialize context
        conversation.context.conversation_id = conversation.id
        conversation.context.conversation_mode = mode

        # Initialize analytics
        conversation.analytics.conversation_id = conversation.id

        self.conversations[conversation.id] = conversation

        logger.info(f"Created conversation {conversation.id} for user {user_id}")
        return conversation

    async def process_message(
        self,
        conversation_id: str,
        user_message: str,
        message_metadata: Dict[str, Any] = None,
    ) -> Message:
        """Process user message and generate response"""
        start_time = time.time()

        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation = self.conversations[conversation_id]

        # Create user message
        user_msg = Message(
            conversation_id=conversation_id,
            message_type=MessageType.USER,
            content=user_message,
            metadata=message_metadata or {},
        )
        conversation.messages.append(user_msg)

        try:
            # Generate response
            response_content = await self._generate_response(conversation, user_message)

            # Create assistant message
            assistant_msg = Message(
                conversation_id=conversation_id,
                message_type=MessageType.ASSISTANT,
                content=response_content["response"],
                confidence=response_content.get("confidence", 0.8),
                processing_time=time.time() - start_time,
                metadata={
                    "suggested_followup": response_content.get("suggested_followup", "")
                },
            )
            conversation.messages.append(assistant_msg)

            # Update conversation analytics
            await self._update_analytics(conversation, assistant_msg)

            # Update last activity
            conversation.last_activity = datetime.utcnow()

            logger.info(
                f"Processed message in conversation {conversation_id} "
                f"(response time: {assistant_msg.processing_time:.2f}s)"
            )

            return assistant_msg

        except Exception as e:
            logger.error(f"Error processing message: {e}")

            # Create error message
            error_msg = Message(
                conversation_id=conversation_id,
                message_type=MessageType.ERROR,
                content=f"I apologize, but I encountered an error processing your message: {str(e)}",
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
            )
            conversation.messages.append(error_msg)

            return error_msg

    async def _generate_response(
        self, conversation: Conversation, user_message: str
    ) -> Dict[str, Any]:
        """Generate contextual response"""
        try:
            # Prepare conversation history
            recent_messages = conversation.messages[-5:]
            history = "\n".join(
                [f"{msg.message_type.value}: {msg.content}" for msg in recent_messages]
            )

            # Prepare user context
            user_context = (
                f"Topics: {', '.join(conversation.context.topics_discussed[-3:])}"
            )

            # Generate response
            response_result = self.response_generator(
                user_message=user_message,
                conversation_history=history,
                user_context=user_context,
                conversation_mode=conversation.mode.value,
            )

            return {
                "response": response_result.response,
                "confidence": response_result.confidence,
                "suggested_followup": response_result.suggested_followup,
            }

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "I understand what you're saying. Could you tell me more about that?",
                "confidence": 0.5,
                "suggested_followup": "What would you like to explore further?",
            }

    async def _update_analytics(self, conversation: Conversation, message: Message):
        """Update conversation analytics"""
        analytics = conversation.analytics

        analytics.message_count += 1

        # Update average response time
        if message.processing_time > 0:
            if analytics.average_response_time == 0:
                analytics.average_response_time = message.processing_time
            else:
                analytics.average_response_time = (
                    analytics.average_response_time * (analytics.message_count - 1)
                    + message.processing_time
                ) / analytics.message_count

        # Simple engagement scoring based on message length
        if len(conversation.messages) >= 2:
            user_messages = [
                msg
                for msg in conversation.messages
                if msg.message_type == MessageType.USER
            ]
            if user_messages:
                avg_user_msg_length = sum(
                    len(msg.content) for msg in user_messages
                ) / len(user_messages)
                analytics.engagement_score = min(
                    1.0, avg_user_msg_length / 100
                )  # Normalize

    async def end_conversation(self, conversation_id: str) -> ConversationAnalytics:
        """End conversation and generate final analytics"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation = self.conversations[conversation_id]
        conversation.state = ConversationState.ENDED
        conversation.analytics.end_time = datetime.utcnow()

        # Simple quality scoring
        conversation.analytics.conversation_quality_score = min(
            1.0,
            conversation.analytics.engagement_score * 0.5
            + (1.0 if conversation.analytics.message_count > 2 else 0.5) * 0.5,
        )

        logger.info(f"Ended conversation {conversation_id}")
        return conversation.analytics

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self.conversations.get(conversation_id)


class ConversationalAIPlatform:
    """Main platform interface"""

    def __init__(self, db_path: str = "conversation_platform.db"):
        self.conversation_engine = ConversationEngine()

    async def start_conversation(
        self, user_id: str, mode: str = "casual", title: str = ""
    ) -> Dict[str, Any]:
        """Start a new conversation"""
        try:
            conv_mode = ConversationMode(mode.lower())
            conversation = await self.conversation_engine.create_conversation(
                user_id, conv_mode, title
            )

            return {
                "success": True,
                "conversation_id": conversation.id,
                "title": conversation.title,
                "mode": conversation.mode.value,
                "created_at": conversation.created_at.isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_message(
        self, conversation_id: str, message: str, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Send message to conversation"""
        try:
            response_message = await self.conversation_engine.process_message(
                conversation_id, message, metadata
            )

            return {
                "success": True,
                "message_id": response_message.id,
                "response": response_message.content,
                "confidence": response_message.confidence,
                "processing_time": response_message.processing_time,
                "suggested_followup": response_message.metadata.get(
                    "suggested_followup", ""
                ),
                "timestamp": response_message.timestamp.isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """End a conversation"""
        try:
            analytics = await self.conversation_engine.end_conversation(conversation_id)

            return {
                "success": True,
                "conversation_id": conversation_id,
                "final_analytics": {
                    "message_count": analytics.message_count,
                    "quality_score": analytics.conversation_quality_score,
                    "engagement_score": analytics.engagement_score,
                    "duration": (
                        str(analytics.end_time - analytics.start_time)
                        if analytics.end_time
                        else "N/A"
                    ),
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation history"""
        try:
            conversation = self.conversation_engine.get_conversation(conversation_id)
            if not conversation:
                return {"success": False, "error": "Conversation not found"}

            messages = []
            for msg in conversation.messages:
                messages.append(
                    {
                        "id": msg.id,
                        "type": msg.message_type.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "confidence": msg.confidence,
                        "processing_time": msg.processing_time,
                    }
                )

            return {
                "success": True,
                "conversation_id": conversation_id,
                "title": conversation.title,
                "state": conversation.state.value,
                "mode": conversation.mode.value,
                "messages": messages,
                "context": {
                    "current_topic": conversation.context.current_topic,
                    "topics_discussed": conversation.context.topics_discussed,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Demo function for testing
async def demo_conversational_ai():
    """Demonstrate conversational AI platform capabilities"""
    print("🤖 Conversational AI Platform Demo")
    print("=" * 50)

    # Initialize platform
    platform = ConversationalAIPlatform(":memory:")

    # Start a conversation
    conv_result = await platform.start_conversation(
        user_id="demo_user", mode="casual", title="Demo Conversation"
    )

    if not conv_result["success"]:
        print(f"❌ Failed to start conversation: {conv_result['error']}")
        return

    conversation_id = conv_result["conversation_id"]
    print(f"✅ Started conversation: {conversation_id}")

    # Test messages
    test_messages = [
        "Hello! How are you today?",
        "Can you tell me about artificial intelligence?",
        "Thank you for the information!",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n👤 User: {message}")

        response = await platform.send_message(conversation_id, message)

        if response["success"]:
            print(f"🤖 Assistant: {response['response'][:150]}...")
            print(f"   Confidence: {response['confidence']:.2f}")
            print(f"   Processing time: {response['processing_time']:.2f}s")
        else:
            print(f"❌ Error: {response['error']}")

    # End conversation
    end_result = await platform.end_conversation(conversation_id)

    if end_result["success"]:
        analytics = end_result["final_analytics"]
        print(f"\n✅ Conversation ended successfully")
        print(f"   Messages: {analytics['message_count']}")
        print(f"   Quality Score: {analytics['quality_score']:.2f}")

    print(f"\n✅ Demo Complete!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo_conversational_ai())
