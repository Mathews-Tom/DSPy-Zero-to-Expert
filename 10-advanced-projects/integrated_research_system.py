#!/usr/bin/env python3
"""
Integrated Multi-Agent Research System

This module integrates all components of the multi-agent research system
into a comprehensive, production-ready research assistant platform.

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent_communication import (
    AdvancedCommunicationManager,
    CommunicationProtocol,
    MessagePriority,
)

# Import our multi-agent components
from multi_agent_system import (
    AgentRole,
    MessageType,
    MultiAgentResearchSystem,
    ResearchContext,
)
from research_workflow import (
    TaskType,
    WorkflowEngine,
    WorkflowStatus,
    WorkflowTemplateManager,
)

logger = logging.getLogger(__name__)


class IntegratedResearchPlatform:
    """Comprehensive integrated research platform"""

    def __init__(self):
        # Core systems
        self.agent_system = MultiAgentResearchSystem()
        self.communication_manager = AdvancedCommunicationManager()
        self.workflow_engine = WorkflowEngine()
        self.template_manager = WorkflowTemplateManager()

        # Platform state
        self.active_projects: Dict[str, Dict[str, Any]] = {}
        self.platform_metrics: Dict[str, Any] = {}
        self.running = False

        # Integration mappings
        self.agent_workflow_mapping: Dict[str, str] = {}  # agent_id -> workflow_id
        self.workflow_communication_mapping: Dict[str, str] = (
            {}
        )  # workflow_id -> conversation_id

    async def initialize_platform(self):
        """Initialize the integrated research platform"""
        logger.info("Initializing Integrated Research Platform...")

        # Initialize core systems
        await self.agent_system.initialize_system()

        # Set up communication topology
        communication_topology = {
            "research_coordination": ["coordinator_001"],
            "research_execution": ["researcher_001", "researcher_002", "analyst_001"],
            "synthesis_evaluation": ["synthesizer_001", "critic_001"],
            "system_wide": list(self.agent_system.agents.keys()),
        }

        await self.communication_manager.setup_communication_topology(
            communication_topology
        )

        logger.info("Platform initialization completed")

    async def start_platform(self):
        """Start the integrated platform"""
        if self.running:
            return

        self.running = True
        logger.info("Starting Integrated Research Platform...")

        # Start agent system
        agent_tasks = await self.agent_system.start_system()

        # Platform is now running
        logger.info("Integrated Research Platform is now running")
        return agent_tasks

    async def stop_platform(self):
        """Stop the integrated platform"""
        self.running = False

        # Stop agent system
        await self.agent_system.stop_system()

        logger.info("Integrated Research Platform stopped")

    async def create_research_project(
        self,
        project_name: str,
        description: str,
        objectives: List[str],
        template_name: str = None,
    ) -> str:
        """Create a comprehensive research project"""
        project_id = str(uuid4())

        logger.info(f"Creating research project: {project_name}")

        # Create workflow
        if template_name:
            # Find template by name
            templates = self.template_manager.list_templates()
            template = next((t for t in templates if t["name"] == template_name), None)

            if template:
                workflow = self.template_manager.create_workflow_from_template(
                    template["id"], project_name, objectives
                )
            else:
                workflow = await self.workflow_engine.create_workflow(
                    project_name, description, objectives
                )
        else:
            workflow = await self.workflow_engine.create_workflow(
                project_name, description, objectives
            )

        # Plan and optimize workflow
        available_agents = list(self.agent_system.agents.keys())
        await self.workflow_engine.plan_workflow(workflow.id, available_agents)

        # Get agent capabilities
        agent_capabilities = {}
        for agent_id, agent in self.agent_system.agents.items():
            agent_capabilities[agent_id] = list(agent.capabilities)

        await self.workflow_engine.optimize_workflow(workflow.id, agent_capabilities)

        # Start conversation for this project
        conversation_id = await self.communication_manager.start_conversation(
            f"project_{project_id}",
            available_agents,
            f"Research Project: {project_name}",
        )

        # Create project record
        self.active_projects[project_id] = {
            "name": project_name,
            "description": description,
            "objectives": objectives,
            "workflow_id": workflow.id,
            "conversation_id": conversation_id,
            "created_at": datetime.utcnow(),
            "status": "created",
            "results": {},
            "metrics": {},
        }

        # Update mappings
        self.workflow_communication_mapping[workflow.id] = conversation_id

        logger.info(
            f"Created research project {project_id} with workflow {workflow.id}"
        )
        return project_id

    async def start_research_project(self, project_id: str) -> bool:
        """Start execution of a research project"""
        if project_id not in self.active_projects:
            return False

        project = self.active_projects[project_id]
        workflow_id = project["workflow_id"]

        # Start workflow execution
        success = await self.workflow_engine.start_workflow(workflow_id)
        if not success:
            return False

        # Start agent-based research
        research_id = await self.agent_system.conduct_research(
            project["name"], project["objectives"]
        )

        # Update project status
        project["status"] = "executing"
        project["started_at"] = datetime.utcnow()
        project["agent_research_id"] = research_id

        # Coordinate workflow and agent execution
        await self._coordinate_project_execution(project_id)

        logger.info(f"Started research project: {project_id}")
        return True

    async def _coordinate_project_execution(self, project_id: str):
        """Coordinate between workflow engine and agent system"""
        project = self.active_projects[project_id]
        workflow_id = project["workflow_id"]

        # Get ready workflow steps
        ready_steps = await self.workflow_engine.get_ready_steps(workflow_id)

        # Assign steps to agents and coordinate execution
        for step in ready_steps:
            if step.assigned_agent:
                # Send task to agent through communication system
                task_message = {
                    "type": "workflow_task",
                    "step_id": step.id,
                    "step_name": step.name,
                    "description": step.description,
                    "task_type": step.task_type.value,
                    "parameters": step.parameters,
                }

                # This would integrate with the agent communication system
                # In a full implementation, this would send actual messages
                logger.info(f"Assigned step {step.name} to agent {step.assigned_agent}")

    async def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive project status"""
        if project_id not in self.active_projects:
            return None

        project = self.active_projects[project_id]

        # Get workflow status
        workflow_status = self.workflow_engine.get_workflow_status(
            project["workflow_id"]
        )

        # Get agent system results
        agent_results = await self.agent_system.get_research_results(
            project.get("agent_research_id", "")
        )

        # Get communication metrics
        comm_metrics = self.communication_manager.get_communication_metrics()

        return {
            "project_id": project_id,
            "name": project["name"],
            "status": project["status"],
            "created_at": project["created_at"].isoformat(),
            "started_at": (
                project.get("started_at", {}).isoformat()
                if project.get("started_at")
                else None
            ),
            "workflow": workflow_status,
            "agent_research": {
                "findings_count": len(agent_results.get("findings", {})),
                "status": agent_results.get("status", "unknown"),
            },
            "communication": {
                "messages_exchanged": comm_metrics.get("total_messages", 0),
                "active_conversations": comm_metrics.get("active_conversations", 0),
            },
            "objectives": project["objectives"],
            "results": project.get("results", {}),
        }

    async def complete_project_step(
        self, project_id: str, step_id: str, results: Dict[str, Any]
    ) -> bool:
        """Complete a project workflow step"""
        if project_id not in self.active_projects:
            return False

        project = self.active_projects[project_id]
        workflow_id = project["workflow_id"]

        # Complete step in workflow engine
        success = await self.workflow_engine.complete_step(
            workflow_id, step_id, results
        )

        if success:
            # Update project results
            if "step_results" not in project["results"]:
                project["results"]["step_results"] = {}
            project["results"]["step_results"][step_id] = results

            # Check if project is complete
            workflow_status = self.workflow_engine.get_workflow_status(workflow_id)
            if workflow_status and workflow_status["status"] == "completed":
                await self._finalize_project(project_id)

        return success

    async def _finalize_project(self, project_id: str):
        """Finalize a completed project"""
        project = self.active_projects[project_id]

        # Get final results from all systems
        workflow_status = self.workflow_engine.get_workflow_status(
            project["workflow_id"]
        )
        agent_results = await self.agent_system.get_research_results(
            project.get("agent_research_id", "")
        )

        # Compile final results
        final_results = {
            "workflow_results": workflow_status,
            "agent_findings": agent_results.get("findings", {}),
            "quality_score": workflow_status.get("quality_score"),
            "completion_time": datetime.utcnow().isoformat(),
            "objectives_met": self._assess_objectives_completion(
                project, agent_results
            ),
        }

        project["results"] = final_results
        project["status"] = "completed"
        project["completed_at"] = datetime.utcnow()

        logger.info(f"Finalized project: {project_id}")

    def _assess_objectives_completion(
        self, project: Dict[str, Any], agent_results: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Assess which objectives were completed"""
        # Simple assessment - in a real system this would be more sophisticated
        objectives_met = {}
        findings_count = len(agent_results.get("findings", {}))

        for i, objective in enumerate(project["objectives"]):
            # Simple heuristic: if we have findings, objectives are likely met
            objectives_met[objective] = findings_count > i

        return objectives_met

    async def generate_project_report(
        self, project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Generate comprehensive project report"""
        if project_id not in self.active_projects:
            return None

        project = self.active_projects[project_id]
        status = await self.get_project_status(project_id)

        if not status:
            return None

        # Generate comprehensive report
        report = {
            "project_summary": {
                "id": project_id,
                "name": project["name"],
                "description": project["description"],
                "status": project["status"],
                "created_at": project["created_at"].isoformat(),
                "duration": self._calculate_project_duration(project),
            },
            "objectives_analysis": {
                "total_objectives": len(project["objectives"]),
                "objectives": project["objectives"],
                "completion_assessment": project.get("results", {}).get(
                    "objectives_met", {}
                ),
            },
            "workflow_performance": status["workflow"],
            "agent_collaboration": {
                "agents_involved": len(self.agent_system.agents),
                "findings_generated": status["agent_research"]["findings_count"],
                "communication_efficiency": status["communication"],
            },
            "quality_metrics": {
                "overall_quality": project.get("results", {}).get("quality_score"),
                "workflow_completion": (
                    status["workflow"]["progress_percent"] if status["workflow"] else 0
                ),
            },
            "key_findings": self._extract_key_findings(project),
            "recommendations": self._generate_recommendations(project),
            "generated_at": datetime.utcnow().isoformat(),
        }

        return report

    def _calculate_project_duration(self, project: Dict[str, Any]) -> Optional[str]:
        """Calculate project duration"""
        if "started_at" not in project:
            return None

        start_time = project["started_at"]
        end_time = project.get("completed_at", datetime.utcnow())

        duration = end_time - start_time
        return str(duration)

    def _extract_key_findings(self, project: Dict[str, Any]) -> List[str]:
        """Extract key findings from project results"""
        findings = []

        # Extract from step results
        step_results = project.get("results", {}).get("step_results", {})
        for step_id, result in step_results.items():
            if "findings" in result:
                findings.append(result["findings"])

        return findings[:5]  # Top 5 findings

    def _generate_recommendations(self, project: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on project results"""
        recommendations = []

        # Basic recommendations based on project status
        if project["status"] == "completed":
            recommendations.append("Consider follow-up research based on findings")
            recommendations.append("Share results with relevant stakeholders")
        else:
            recommendations.append("Continue monitoring project progress")
            recommendations.append("Address any identified bottlenecks")

        return recommendations

    def get_platform_metrics(self) -> Dict[str, Any]:
        """Get comprehensive platform metrics"""
        agent_status = self.agent_system.get_system_status()
        workflow_metrics = self.workflow_engine.get_system_metrics()
        comm_metrics = self.communication_manager.get_communication_metrics()

        return {
            "platform_status": {
                "running": self.running,
                "active_projects": len(self.active_projects),
                "total_projects": len(
                    self.active_projects
                ),  # In full system, would include completed
            },
            "agent_system": agent_status,
            "workflow_system": workflow_metrics,
            "communication_system": comm_metrics,
            "integration_metrics": {
                "agent_workflow_mappings": len(self.agent_workflow_mapping),
                "workflow_communication_mappings": len(
                    self.workflow_communication_mapping
                ),
            },
        }

    async def list_available_templates(self) -> List[Dict[str, Any]]:
        """List available workflow templates"""
        return self.template_manager.list_templates()


async def demonstrate_integrated_platform():
    """Demonstrate the integrated research platform"""
    print("=== Integrated Multi-Agent Research Platform Demo ===")

    # Initialize platform
    platform = IntegratedResearchPlatform()
    await platform.initialize_platform()

    # Start platform
    print("\nStarting integrated platform...")
    agent_tasks = await platform.start_platform()

    # Wait for system stabilization
    await asyncio.sleep(2)

    # List available templates
    print("\nAvailable workflow templates:")
    templates = await platform.list_available_templates()
    for template in templates:
        print(f"  - {template['name']}: {template['description']}")

    # Create research project
    print("\nCreating research project...")
    project_id = await platform.create_research_project(
        project_name="AI Ethics in Healthcare",
        description="Comprehensive analysis of ethical considerations in healthcare AI",
        objectives=[
            "Identify key ethical challenges in healthcare AI",
            "Analyze current regulatory frameworks",
            "Propose ethical guidelines for AI implementation",
            "Assess stakeholder perspectives",
        ],
        template_name="Literature Review",
    )

    print(f"Created project: {project_id}")

    # Start project execution
    print("\nStarting project execution...")
    success = await platform.start_research_project(project_id)
    print(f"Project started: {'Success' if success else 'Failed'}")

    # Monitor project progress
    print("\nMonitoring project progress...")
    for i in range(5):
        await asyncio.sleep(2)

        status = await platform.get_project_status(project_id)
        if status:
            print(f"  Update {i+1}: {status['workflow']['progress']} steps completed")
            print(f"    Agent findings: {status['agent_research']['findings_count']}")
            print(
                f"    Messages exchanged: {status['communication']['messages_exchanged']}"
            )

    # Simulate step completion
    print("\nSimulating step completions...")
    status = await platform.get_project_status(project_id)
    if status and status["workflow"]["steps"]:
        for step in status["workflow"]["steps"][:2]:  # Complete first 2 steps
            await platform.complete_project_step(
                project_id,
                step["id"],
                {
                    "findings": f"Completed findings for {step['name']}",
                    "confidence": 0.85,
                    "sources": ["source1", "source2"],
                },
            )
            print(f"  Completed step: {step['name']}")

    # Generate project report
    print("\nGenerating project report...")
    report = await platform.generate_project_report(project_id)

    if report:
        print(f"\nProject Report Summary:")
        print(f"  Project: {report['project_summary']['name']}")
        print(f"  Status: {report['project_summary']['status']}")
        print(f"  Objectives: {report['objectives_analysis']['total_objectives']}")
        print(f"  Quality Score: {report['quality_metrics']['overall_quality']}")
        print(f"  Key Findings: {len(report['key_findings'])}")
        print(f"  Recommendations: {len(report['recommendations'])}")

    # Get platform metrics
    print("\nPlatform Metrics:")
    metrics = platform.get_platform_metrics()
    print(f"  Active Projects: {metrics['platform_status']['active_projects']}")
    print(f"  Active Agents: {metrics['agent_system']['active_agents']}")
    print(f"  Completed Workflows: {metrics['workflow_system']['completed_workflows']}")
    print(f"  Total Messages: {metrics['communication_system']['total_messages']}")

    # Stop platform
    print("\nStopping integrated platform...")
    await platform.stop_platform()

    # Cancel agent tasks
    for task in agent_tasks:
        task.cancel()

    print("Integrated platform demo completed!")


async def main():
    """Main demonstration function"""
    try:
        await demonstrate_integrated_platform()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
