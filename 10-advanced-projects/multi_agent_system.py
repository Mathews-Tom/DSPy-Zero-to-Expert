#!/usr/bin/env python3
"""
Multi-Agent Research Assistant System

This module implements a sophisticated multi-agent system for collaborative research
using DSPy. The system features specialized agents that work together to conduct
comprehensive research, analyze information, and generate insights.

Learning Objectives:
- Implement multi-agent architectures with DSPy
- Create agent coordination and communication frameworks
- Build research workflow automation systems
- Develop collaborative intelligence patterns
- Master advanced DSPy composition techniques

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

import dspy
from dspy import ChainOfThought, Predict, Retrieve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in the research system"""

    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    SPECIALIST = "specialist"


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Inter-agent message types"""

    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    INFORMATION_SHARE = "information_share"
    COORDINATION = "coordination"
    QUERY = "query"
    RESULT = "result"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""

    id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.INFORMATION_SHARE
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 1  # 1-10, 10 being highest
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class ResearchTask:
    """Research task structure"""

    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchContext:
    """Shared research context"""

    topic: str = ""
    objectives: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    findings: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# DSPy Signatures for Multi-Agent System
class ResearchQuery(dspy.Signature):
    """Generate research queries for a given topic"""

    topic: str = dspy.InputField(desc="Research topic or question")
    context: str = dspy.InputField(desc="Additional context or constraints")
    queries: List[str] = dspy.OutputField(desc="List of specific research queries")


class InformationAnalysis(dspy.Signature):
    """Analyze and extract key insights from research information"""

    information: str = dspy.InputField(desc="Raw research information")
    context: str = dspy.InputField(desc="Research context and objectives")
    key_insights: str = dspy.OutputField(desc="Key insights and findings")
    relevance_score: float = dspy.OutputField(desc="Relevance score (0-1)")
    confidence: float = dspy.OutputField(desc="Confidence in analysis (0-1)")


class SynthesisGeneration(dspy.Signature):
    """Synthesize multiple research findings into coherent insights"""

    findings: str = dspy.InputField(desc="Multiple research findings")
    objectives: str = dspy.InputField(desc="Research objectives")
    synthesis: str = dspy.OutputField(desc="Synthesized insights and conclusions")
    recommendations: str = dspy.OutputField(desc="Actionable recommendations")


class CriticalEvaluation(dspy.Signature):
    """Critically evaluate research findings and synthesis"""

    content: str = dspy.InputField(desc="Content to evaluate")
    criteria: str = dspy.InputField(desc="Evaluation criteria")
    evaluation: str = dspy.OutputField(desc="Critical evaluation and feedback")
    strengths: str = dspy.OutputField(desc="Identified strengths")
    weaknesses: str = dspy.OutputField(desc="Identified weaknesses")
    suggestions: str = dspy.OutputField(desc="Improvement suggestions")


class TaskCoordination(dspy.Signature):
    """Coordinate task assignment and workflow"""

    available_agents: str = dspy.InputField(
        desc="Available agents and their capabilities"
    )
    pending_tasks: str = dspy.InputField(desc="Pending tasks to be assigned")
    current_context: str = dspy.InputField(desc="Current research context")
    task_assignments: str = dspy.OutputField(desc="Optimal task assignments")
    workflow_plan: str = dspy.OutputField(desc="Workflow execution plan")


class MessageBus:
    """Central message bus for inter-agent communication"""

    def __init__(self):
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, List[str]] = {}  # message_type -> agent_ids
        self.message_handlers: Dict[str, callable] = {}
        self._lock = asyncio.Lock()

    async def publish(self, message: AgentMessage):
        """Publish a message to the bus"""
        async with self._lock:
            self.messages.append(message)
            logger.debug(f"Published message {message.id} from {message.sender_id}")

    async def subscribe(self, agent_id: str, message_types: List[MessageType]):
        """Subscribe an agent to specific message types"""
        async with self._lock:
            for msg_type in message_types:
                if msg_type.value not in self.subscribers:
                    self.subscribers[msg_type.value] = []
                if agent_id not in self.subscribers[msg_type.value]:
                    self.subscribers[msg_type.value].append(agent_id)

    async def get_messages(
        self, agent_id: str, since: Optional[datetime] = None
    ) -> List[AgentMessage]:
        """Get messages for a specific agent"""
        async with self._lock:
            messages = []
            for message in self.messages:
                # Check if message is for this agent
                if (
                    message.recipient_id == agent_id
                    or message.recipient_id == "all"
                    or any(
                        agent_id in self.subscribers.get(message.message_type.value, [])
                    )
                ):

                    if since is None or message.timestamp > since:
                        messages.append(message)

            return sorted(messages, key=lambda m: m.timestamp)


class BaseAgent(ABC):
    """Base class for all agents in the system"""

    def __init__(self, agent_id: str, role: AgentRole, message_bus: MessageBus):
        self.agent_id = agent_id
        self.role = role
        self.message_bus = message_bus
        self.capabilities: Set[str] = set()
        self.active = True
        self.last_message_check = datetime.utcnow()
        self.context: Dict[str, Any] = {}

        # Initialize DSPy modules
        self._init_dspy_modules()

    @abstractmethod
    def _init_dspy_modules(self):
        """Initialize DSPy modules specific to this agent"""
        pass

    @abstractmethod
    async def process_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process a research task"""
        pass

    async def send_message(
        self,
        recipient_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        requires_response: bool = False,
    ):
        """Send a message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            requires_response=requires_response,
        )
        await self.message_bus.publish(message)

    async def broadcast_message(
        self, message_type: MessageType, content: Dict[str, Any]
    ):
        """Broadcast a message to all agents"""
        await self.send_message("all", message_type, content)

    async def check_messages(self) -> List[AgentMessage]:
        """Check for new messages"""
        messages = await self.message_bus.get_messages(
            self.agent_id, self.last_message_check
        )
        self.last_message_check = datetime.utcnow()
        return messages

    async def handle_message(self, message: AgentMessage):
        """Handle an incoming message"""
        logger.debug(f"Agent {self.agent_id} handling message {message.id}")

        if message.message_type == MessageType.TASK_REQUEST:
            await self._handle_task_request(message)
        elif message.message_type == MessageType.INFORMATION_SHARE:
            await self._handle_information_share(message)
        elif message.message_type == MessageType.COORDINATION:
            await self._handle_coordination(message)
        elif message.message_type == MessageType.QUERY:
            await self._handle_query(message)

    async def _handle_task_request(self, message: AgentMessage):
        """Handle task request message"""
        task_data = message.content.get("task")
        if task_data:
            task = ResearchTask(**task_data)
            result = await self.process_task(task)

            # Send response
            await self.send_message(
                message.sender_id,
                MessageType.TASK_RESPONSE,
                {"task_id": task.id, "result": result},
            )

    async def _handle_information_share(self, message: AgentMessage):
        """Handle information sharing message"""
        info = message.content.get("information")
        if info:
            # Store information in context
            self.context[f"shared_info_{message.id}"] = info

    async def _handle_coordination(self, message: AgentMessage):
        """Handle coordination message"""
        # Default coordination handling
        pass

    async def _handle_query(self, message: AgentMessage):
        """Handle query message"""
        query = message.content.get("query")
        if query and message.requires_response:
            # Process query and send response
            response = await self._process_query(query)
            await self.send_message(
                message.sender_id,
                MessageType.RESULT,
                {"query": query, "response": response},
            )

    async def _process_query(self, query: str) -> str:
        """Process a query - to be overridden by specific agents"""
        return f"Agent {self.agent_id} received query: {query}"

    async def run(self):
        """Main agent execution loop"""
        logger.info(f"Starting agent {self.agent_id} ({self.role.value})")

        while self.active:
            try:
                # Check for messages
                messages = await self.check_messages()

                # Process messages
                for message in messages:
                    await self.handle_message(message)

                # Agent-specific processing
                await self._agent_specific_processing()

                # Brief pause
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in agent {self.agent_id}: {e}")
                await asyncio.sleep(5)

    async def _agent_specific_processing(self):
        """Agent-specific processing - to be overridden"""
        pass


class CoordinatorAgent(BaseAgent):
    """Coordinator agent that manages research workflow"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, AgentRole.COORDINATOR, message_bus)
        self.active_tasks: Dict[str, ResearchTask] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.research_context = ResearchContext()

    def _init_dspy_modules(self):
        """Initialize DSPy modules for coordination"""
        self.task_coordinator = ChainOfThought(TaskCoordination)
        self.query_generator = ChainOfThought(ResearchQuery)

    async def start_research(self, topic: str, objectives: List[str]) -> str:
        """Start a new research project"""
        research_id = str(uuid4())

        self.research_context = ResearchContext(topic=topic, objectives=objectives)

        # Generate initial research queries
        context_str = f"Objectives: {', '.join(objectives)}"
        queries_result = self.query_generator(topic=topic, context=context_str)

        # Create research tasks
        tasks = []
        for i, query in enumerate(queries_result.queries):
            task = ResearchTask(
                title=f"Research Query {i+1}", description=query, priority=5
            )
            tasks.append(task)
            self.active_tasks[task.id] = task

        # Coordinate task assignment
        await self._coordinate_tasks(tasks)

        logger.info(f"Started research project: {topic}")
        return research_id

    async def _coordinate_tasks(self, tasks: List[ResearchTask]):
        """Coordinate task assignment among agents"""
        # Get available agents
        available_agents = list(self.agent_capabilities.keys())

        if not available_agents:
            logger.warning("No agents available for task assignment")
            return

        # Prepare coordination input
        agents_str = json.dumps(
            {
                agent_id: list(capabilities)
                for agent_id, capabilities in self.agent_capabilities.items()
            }
        )

        tasks_str = json.dumps(
            [
                {"id": task.id, "title": task.title, "description": task.description}
                for task in tasks
            ]
        )

        context_str = json.dumps(
            {
                "topic": self.research_context.topic,
                "objectives": self.research_context.objectives,
            }
        )

        # Get task assignments
        coordination_result = self.task_coordinator(
            available_agents=agents_str,
            pending_tasks=tasks_str,
            current_context=context_str,
        )

        # Parse and assign tasks
        try:
            assignments = json.loads(coordination_result.task_assignments)
            for assignment in assignments:
                task_id = assignment.get("task_id")
                agent_id = assignment.get("agent_id")

                if task_id in self.active_tasks and agent_id in self.agent_capabilities:
                    task = self.active_tasks[task_id]
                    task.assigned_agent = agent_id
                    task.status = TaskStatus.IN_PROGRESS

                    # Send task to agent
                    await self.send_message(
                        agent_id, MessageType.TASK_REQUEST, {"task": task.__dict__}
                    )
        except json.JSONDecodeError:
            logger.error("Failed to parse task assignments")

    async def register_agent(self, agent_id: str, capabilities: Set[str]):
        """Register an agent and its capabilities"""
        self.agent_capabilities[agent_id] = capabilities
        logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")

    async def process_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process coordination tasks"""
        return {"status": "coordination_complete", "message": "Task coordinated"}

    async def _handle_task_response(self, message: AgentMessage):
        """Handle task completion responses"""
        task_id = message.content.get("task_id")
        result = message.content.get("result")

        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.results = result

            # Store findings in research context
            self.research_context.findings[task_id] = result

            logger.info(f"Task {task_id} completed by {message.sender_id}")


class ResearcherAgent(BaseAgent):
    """Researcher agent that conducts information gathering"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, AgentRole.RESEARCHER, message_bus)
        self.capabilities = {"research", "information_gathering", "web_search"}

    def _init_dspy_modules(self):
        """Initialize DSPy modules for research"""
        self.information_analyzer = ChainOfThought(InformationAnalysis)
        # Note: In a real implementation, you'd set up retrieval modules
        # self.retriever = Retrieve(k=5)

    async def process_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process research tasks"""
        logger.info(f"Researcher {self.agent_id} processing task: {task.title}")

        # Simulate research process
        await asyncio.sleep(2)  # Simulate research time

        # In a real implementation, this would involve:
        # 1. Web search using the retriever
        # 2. Document analysis
        # 3. Information extraction

        # Simulated research results
        research_info = f"Research findings for: {task.description}"
        context = f"Task: {task.title}"

        # Analyze the information
        analysis_result = self.information_analyzer(
            information=research_info, context=context
        )

        return {
            "findings": analysis_result.key_insights,
            "relevance_score": analysis_result.relevance_score,
            "confidence": analysis_result.confidence,
            "sources": ["simulated_source_1", "simulated_source_2"],
            "timestamp": datetime.utcnow().isoformat(),
        }


class AnalystAgent(BaseAgent):
    """Analyst agent that processes and analyzes research data"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, AgentRole.ANALYST, message_bus)
        self.capabilities = {"analysis", "data_processing", "pattern_recognition"}

    def _init_dspy_modules(self):
        """Initialize DSPy modules for analysis"""
        self.information_analyzer = ChainOfThought(InformationAnalysis)

    async def process_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process analysis tasks"""
        logger.info(f"Analyst {self.agent_id} processing task: {task.title}")

        # Simulate analysis process
        await asyncio.sleep(1.5)

        # Analyze the task content
        analysis_result = self.information_analyzer(
            information=task.description, context=f"Analysis task: {task.title}"
        )

        return {
            "analysis": analysis_result.key_insights,
            "patterns": ["pattern_1", "pattern_2"],
            "insights": "Key analytical insights",
            "confidence": analysis_result.confidence,
            "timestamp": datetime.utcnow().isoformat(),
        }


class SynthesizerAgent(BaseAgent):
    """Synthesizer agent that combines findings into coherent insights"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, AgentRole.SYNTHESIZER, message_bus)
        self.capabilities = {"synthesis", "integration", "report_generation"}

    def _init_dspy_modules(self):
        """Initialize DSPy modules for synthesis"""
        self.synthesizer = ChainOfThought(SynthesisGeneration)

    async def process_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process synthesis tasks"""
        logger.info(f"Synthesizer {self.agent_id} processing task: {task.title}")

        # Simulate synthesis process
        await asyncio.sleep(2.5)

        # Synthesize findings
        findings_str = f"Multiple research findings: {task.description}"
        objectives_str = "Research objectives and goals"

        synthesis_result = self.synthesizer(
            findings=findings_str, objectives=objectives_str
        )

        return {
            "synthesis": synthesis_result.synthesis,
            "recommendations": synthesis_result.recommendations,
            "key_themes": ["theme_1", "theme_2", "theme_3"],
            "confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat(),
        }


class CriticAgent(BaseAgent):
    """Critic agent that evaluates and provides feedback"""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, AgentRole.CRITIC, message_bus)
        self.capabilities = {"evaluation", "quality_assessment", "feedback"}

    def _init_dspy_modules(self):
        """Initialize DSPy modules for criticism"""
        self.evaluator = ChainOfThought(CriticalEvaluation)

    async def process_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process evaluation tasks"""
        logger.info(f"Critic {self.agent_id} processing task: {task.title}")

        # Simulate evaluation process
        await asyncio.sleep(1)

        # Evaluate content
        evaluation_result = self.evaluator(
            content=task.description,
            criteria="Research quality, accuracy, completeness",
        )

        return {
            "evaluation": evaluation_result.evaluation,
            "strengths": evaluation_result.strengths,
            "weaknesses": evaluation_result.weaknesses,
            "suggestions": evaluation_result.suggestions,
            "quality_score": 0.8,
            "timestamp": datetime.utcnow().isoformat(),
        }


class MultiAgentResearchSystem:
    """Main multi-agent research system orchestrator"""

    def __init__(self):
        self.message_bus = MessageBus()
        self.agents: Dict[str, BaseAgent] = {}
        self.coordinator: Optional[CoordinatorAgent] = None
        self.research_sessions: Dict[str, ResearchContext] = {}
        self.running = False

    async def initialize_system(self):
        """Initialize the multi-agent system"""
        logger.info("Initializing Multi-Agent Research System...")

        # Create coordinator
        self.coordinator = CoordinatorAgent("coordinator_001", self.message_bus)
        self.agents["coordinator_001"] = self.coordinator

        # Create specialized agents
        agents_config = [
            ("researcher_001", ResearcherAgent),
            ("researcher_002", ResearcherAgent),
            ("analyst_001", AnalystAgent),
            ("synthesizer_001", SynthesizerAgent),
            ("critic_001", CriticAgent),
        ]

        for agent_id, agent_class in agents_config:
            agent = agent_class(agent_id, self.message_bus)
            self.agents[agent_id] = agent

            # Register agent with coordinator
            await self.coordinator.register_agent(agent_id, agent.capabilities)

        # Set up message subscriptions
        for agent_id, agent in self.agents.items():
            await self.message_bus.subscribe(
                agent_id,
                [
                    MessageType.TASK_REQUEST,
                    MessageType.TASK_RESPONSE,
                    MessageType.INFORMATION_SHARE,
                    MessageType.COORDINATION,
                    MessageType.QUERY,
                ],
            )

        logger.info(f"Initialized {len(self.agents)} agents")

    async def start_system(self):
        """Start the multi-agent system"""
        if self.running:
            return

        self.running = True
        logger.info("Starting Multi-Agent Research System...")

        # Start all agents
        agent_tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(agent.run())
            agent_tasks.append(task)

        return agent_tasks

    async def stop_system(self):
        """Stop the multi-agent system"""
        self.running = False

        # Stop all agents
        for agent in self.agents.values():
            agent.active = False

        logger.info("Multi-Agent Research System stopped")

    async def conduct_research(self, topic: str, objectives: List[str]) -> str:
        """Conduct research using the multi-agent system"""
        if not self.coordinator:
            raise ValueError("System not initialized")

        logger.info(f"Starting research on topic: {topic}")

        # Start research project
        research_id = await self.coordinator.start_research(topic, objectives)

        return research_id

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "running": self.running,
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.active]),
            "agent_roles": {
                agent_id: agent.role.value for agent_id, agent in self.agents.items()
            },
            "message_count": len(self.message_bus.messages),
            "research_sessions": len(self.research_sessions),
        }

    async def get_research_results(self, research_id: str) -> Dict[str, Any]:
        """Get results from a research session"""
        if not self.coordinator:
            return {"error": "System not initialized"}

        context = self.coordinator.research_context

        return {
            "research_id": research_id,
            "topic": context.topic,
            "objectives": context.objectives,
            "findings": context.findings,
            "sources": context.sources,
            "timeline": context.timeline,
            "status": "completed" if context.findings else "in_progress",
        }


async def demonstrate_multi_agent_system():
    """Demonstrate the multi-agent research system"""
    print("=== Multi-Agent Research Assistant System Demo ===")

    # Initialize system
    system = MultiAgentResearchSystem()
    await system.initialize_system()

    # Start system
    print("\nStarting multi-agent system...")
    agent_tasks = await system.start_system()

    # Wait for system to stabilize
    await asyncio.sleep(2)

    # Conduct research
    print("\nConducting research on 'Artificial Intelligence in Healthcare'...")
    research_id = await system.conduct_research(
        topic="Artificial Intelligence in Healthcare",
        objectives=[
            "Identify current AI applications in healthcare",
            "Analyze benefits and challenges",
            "Explore future opportunities",
            "Assess ethical considerations",
        ],
    )

    print(f"Research session started: {research_id}")

    # Monitor system status
    print("\nMonitoring system status...")
    for i in range(10):
        status = system.get_system_status()
        print(
            f"Status update {i+1}: {status['active_agents']}/{status['total_agents']} agents active, "
            f"{status['message_count']} messages"
        )
        await asyncio.sleep(3)

    # Get research results
    print("\nRetrieving research results...")
    results = await system.get_research_results(research_id)

    print(f"\nResearch Results:")
    print(f"Topic: {results['topic']}")
    print(f"Objectives: {len(results['objectives'])} objectives")
    print(f"Findings: {len(results['findings'])} findings collected")
    print(f"Status: {results['status']}")

    if results["findings"]:
        print("\nKey Findings:")
        for task_id, finding in list(results["findings"].items())[:3]:
            print(f"- {finding.get('findings', 'Processing...')}")

    # Stop system
    print("\nStopping multi-agent system...")
    await system.stop_system()

    # Cancel agent tasks
    for task in agent_tasks:
        task.cancel()

    print("Demo completed!")


async def main():
    """Main demonstration function"""
    # Set up a simple DSPy configuration for demo
    # In a real implementation, you'd configure with actual LM and retrieval

    try:
        await demonstrate_multi_agent_system()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
