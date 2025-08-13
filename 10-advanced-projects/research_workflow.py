#!/usr/bin/env python3
"""
Research Workflow Management System

This module provides advanced workflow management capabilities for
multi-agent research systems using DSPy.

Author: DSPy Learning Framework
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import dspy

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""

    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of research tasks"""

    INFORMATION_GATHERING = "information_gathering"
    DATA_ANALYSIS = "data_analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    REPORTING = "reporting"
    COORDINATION = "coordination"


class DependencyType(Enum):
    """Types of task dependencies"""

    SEQUENTIAL = "sequential"  # Must complete before next starts
    PARALLEL = "parallel"  # Can run simultaneously
    CONDITIONAL = "conditional"  # Depends on condition
    RESOURCE = "resource"  # Depends on resource availability


@dataclass
class WorkflowStep:
    """Individual step in a research workflow"""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    task_type: TaskType = TaskType.INFORMATION_GATHERING
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    dependency_type: DependencyType = DependencyType.SEQUENTIAL
    estimated_duration: int = 60  # minutes
    priority: int = 5  # 1-10
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ResearchWorkflow:
    """Complete research workflow definition"""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    objectives: List[str] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTemplate:
    """Template for creating research workflows"""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""
    step_templates: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


# DSPy Signatures for Workflow Management
class WorkflowPlanning(dspy.Signature):
    """Plan research workflow based on objectives"""

    research_topic: str = dspy.InputField(desc="Main research topic")
    objectives: str = dspy.InputField(desc="Research objectives")
    constraints: str = dspy.InputField(desc="Time, resource, and other constraints")
    available_agents: str = dspy.InputField(desc="Available agents and capabilities")
    workflow_plan: str = dspy.OutputField(desc="Detailed workflow plan with steps")
    estimated_timeline: str = dspy.OutputField(desc="Estimated timeline for completion")


class TaskOptimization(dspy.Signature):
    """Optimize task assignment and scheduling"""

    workflow_steps: str = dspy.InputField(desc="Workflow steps to optimize")
    agent_capabilities: str = dspy.InputField(
        desc="Agent capabilities and availability"
    )
    resource_constraints: str = dspy.InputField(desc="Resource constraints")
    optimized_assignment: str = dspy.OutputField(desc="Optimized task assignments")
    efficiency_improvements: str = dspy.OutputField(
        desc="Expected efficiency improvements"
    )


class WorkflowAdaptation(dspy.Signature):
    """Adapt workflow based on intermediate results"""

    current_workflow: str = dspy.InputField(desc="Current workflow state")
    intermediate_results: str = dspy.InputField(desc="Results from completed steps")
    new_insights: str = dspy.InputField(desc="New insights or changed requirements")
    adapted_workflow: str = dspy.OutputField(desc="Adapted workflow plan")
    adaptation_rationale: str = dspy.OutputField(desc="Rationale for adaptations")


class QualityAssessment(dspy.Signature):
    """Assess quality of workflow execution and results"""

    workflow_results: str = dspy.InputField(desc="Complete workflow results")
    original_objectives: str = dspy.InputField(desc="Original research objectives")
    quality_criteria: str = dspy.InputField(desc="Quality assessment criteria")
    quality_score: float = dspy.OutputField(desc="Overall quality score (0-1)")
    quality_report: str = dspy.OutputField(desc="Detailed quality assessment")
    improvement_suggestions: str = dspy.OutputField(desc="Suggestions for improvement")


class WorkflowEngine:
    """Core workflow execution engine"""

    def __init__(self):
        self.active_workflows: Dict[str, ResearchWorkflow] = {}
        self.workflow_history: List[ResearchWorkflow] = []
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.agent_assignments: Dict[str, Set[str]] = {}  # agent_id -> step_ids
        self.step_dependencies: Dict[str, Set[str]] = (
            {}
        )  # step_id -> dependency_step_ids

        # Initialize DSPy modules
        self.workflow_planner = dspy.ChainOfThought(WorkflowPlanning)
        self.task_optimizer = dspy.ChainOfThought(TaskOptimization)
        self.workflow_adapter = dspy.ChainOfThought(WorkflowAdaptation)
        self.quality_assessor = dspy.ChainOfThought(QualityAssessment)

    async def create_workflow(
        self, name: str, description: str, objectives: List[str]
    ) -> ResearchWorkflow:
        """Create a new research workflow"""
        workflow = ResearchWorkflow(
            name=name, description=description, objectives=objectives
        )

        self.active_workflows[workflow.id] = workflow
        logger.info(f"Created workflow: {workflow.name} ({workflow.id})")

        return workflow

    async def plan_workflow(
        self,
        workflow_id: str,
        available_agents: List[str],
        constraints: Dict[str, Any] = None,
    ) -> bool:
        """Plan workflow steps using DSPy"""
        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.PLANNING

        # Prepare inputs for planning
        objectives_str = "; ".join(workflow.objectives)
        constraints_str = json.dumps(constraints or {})
        agents_str = json.dumps(available_agents)

        # Generate workflow plan
        planning_result = self.workflow_planner(
            research_topic=workflow.description,
            objectives=objectives_str,
            constraints=constraints_str,
            available_agents=agents_str,
        )

        # Parse and create workflow steps
        try:
            plan_data = json.loads(planning_result.workflow_plan)
            steps = []

            for step_data in plan_data.get("steps", []):
                step = WorkflowStep(
                    name=step_data.get("name", ""),
                    description=step_data.get("description", ""),
                    task_type=TaskType(
                        step_data.get("task_type", "information_gathering")
                    ),
                    estimated_duration=step_data.get("duration", 60),
                    priority=step_data.get("priority", 5),
                    dependencies=step_data.get("dependencies", []),
                )
                steps.append(step)

            workflow.steps = steps
            workflow.metadata["planning_result"] = planning_result.workflow_plan
            workflow.metadata["estimated_timeline"] = planning_result.estimated_timeline

            logger.info(f"Planned workflow {workflow_id} with {len(steps)} steps")
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse workflow plan: {e}")
            return False

    async def optimize_workflow(
        self, workflow_id: str, agent_capabilities: Dict[str, List[str]]
    ) -> bool:
        """Optimize workflow task assignments"""
        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]

        # Prepare optimization inputs
        steps_data = []
        for step in workflow.steps:
            steps_data.append(
                {
                    "id": step.id,
                    "name": step.name,
                    "task_type": step.task_type.value,
                    "duration": step.estimated_duration,
                    "dependencies": step.dependencies,
                }
            )

        steps_str = json.dumps(steps_data)
        capabilities_str = json.dumps(agent_capabilities)

        # Optimize assignments
        optimization_result = self.task_optimizer(
            workflow_steps=steps_str,
            agent_capabilities=capabilities_str,
            resource_constraints="{}",
        )

        # Apply optimized assignments
        try:
            assignments = json.loads(optimization_result.optimized_assignment)

            for assignment in assignments:
                step_id = assignment.get("step_id")
                agent_id = assignment.get("agent_id")

                # Find and update step
                for step in workflow.steps:
                    if step.id == step_id:
                        step.assigned_agent = agent_id
                        break

                # Update agent assignments tracking
                if agent_id not in self.agent_assignments:
                    self.agent_assignments[agent_id] = set()
                self.agent_assignments[agent_id].add(step_id)

            workflow.metadata["optimization_result"] = (
                optimization_result.optimized_assignment
            )
            workflow.metadata["efficiency_improvements"] = (
                optimization_result.efficiency_improvements
            )

            logger.info(f"Optimized workflow {workflow_id} assignments")
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse optimization result: {e}")
            return False

    async def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution"""
        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.EXECUTING
        workflow.started_at = datetime.utcnow()

        # Build dependency graph
        self._build_dependency_graph(workflow)

        logger.info(f"Started workflow execution: {workflow_id}")
        return True

    def _build_dependency_graph(self, workflow: ResearchWorkflow):
        """Build dependency graph for workflow steps"""
        for step in workflow.steps:
            self.step_dependencies[step.id] = set(step.dependencies)

    async def get_ready_steps(self, workflow_id: str) -> List[WorkflowStep]:
        """Get steps that are ready to execute"""
        if workflow_id not in self.active_workflows:
            return []

        workflow = self.active_workflows[workflow_id]
        ready_steps = []

        for step in workflow.steps:
            if step.status == "pending":
                # Check if all dependencies are completed
                dependencies_met = True
                for dep_id in step.dependencies:
                    dep_step = self._find_step_by_id(workflow, dep_id)
                    if not dep_step or dep_step.status != "completed":
                        dependencies_met = False
                        break

                if dependencies_met:
                    ready_steps.append(step)

        return ready_steps

    def _find_step_by_id(
        self, workflow: ResearchWorkflow, step_id: str
    ) -> Optional[WorkflowStep]:
        """Find step by ID in workflow"""
        for step in workflow.steps:
            if step.id == step_id:
                return step
        return None

    async def complete_step(
        self, workflow_id: str, step_id: str, results: Dict[str, Any]
    ) -> bool:
        """Mark a step as completed with results"""
        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]
        step = self._find_step_by_id(workflow, step_id)

        if not step:
            return False

        step.status = "completed"
        step.completed_at = datetime.utcnow()
        step.results = results

        # Check if workflow is complete
        all_completed = all(s.status == "completed" for s in workflow.steps)
        if all_completed:
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()

            # Assess workflow quality
            await self._assess_workflow_quality(workflow)

            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]

            logger.info(f"Workflow {workflow_id} completed")

        return True

    async def _assess_workflow_quality(self, workflow: ResearchWorkflow):
        """Assess the quality of completed workflow"""
        # Compile results
        results_data = {}
        for step in workflow.steps:
            results_data[step.id] = step.results

        results_str = json.dumps(results_data)
        objectives_str = "; ".join(workflow.objectives)

        # Assess quality
        quality_result = self.quality_assessor(
            workflow_results=results_str,
            original_objectives=objectives_str,
            quality_criteria="Completeness, accuracy, relevance, timeliness",
        )

        workflow.metrics["quality_score"] = quality_result.quality_score
        workflow.metadata["quality_report"] = quality_result.quality_report
        workflow.metadata["improvement_suggestions"] = (
            quality_result.improvement_suggestions
        )

    async def adapt_workflow(self, workflow_id: str, new_insights: str) -> bool:
        """Adapt workflow based on intermediate results"""
        if workflow_id not in self.active_workflows:
            return False

        workflow = self.active_workflows[workflow_id]

        # Compile intermediate results
        intermediate_results = {}
        for step in workflow.steps:
            if step.status == "completed":
                intermediate_results[step.id] = step.results

        current_workflow_str = json.dumps(
            {
                "steps": [
                    {"id": s.id, "name": s.name, "status": s.status}
                    for s in workflow.steps
                ]
            }
        )

        results_str = json.dumps(intermediate_results)

        # Get adaptation suggestions
        adaptation_result = self.workflow_adapter(
            current_workflow=current_workflow_str,
            intermediate_results=results_str,
            new_insights=new_insights,
        )

        # Store adaptation information
        workflow.metadata["adaptation_suggestions"] = adaptation_result.adapted_workflow
        workflow.metadata["adaptation_rationale"] = (
            adaptation_result.adaptation_rationale
        )

        logger.info(f"Generated adaptation suggestions for workflow {workflow_id}")
        return True

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
        else:
            # Check history
            workflow = next(
                (w for w in self.workflow_history if w.id == workflow_id), None
            )
            if not workflow:
                return None

        completed_steps = len([s for s in workflow.steps if s.status == "completed"])
        total_steps = len(workflow.steps)

        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress": f"{completed_steps}/{total_steps}",
            "progress_percent": (
                (completed_steps / total_steps * 100) if total_steps > 0 else 0
            ),
            "created_at": workflow.created_at.isoformat(),
            "started_at": (
                workflow.started_at.isoformat() if workflow.started_at else None
            ),
            "completed_at": (
                workflow.completed_at.isoformat() if workflow.completed_at else None
            ),
            "quality_score": workflow.metrics.get("quality_score"),
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "status": step.status,
                    "assigned_agent": step.assigned_agent,
                }
                for step in workflow.steps
            ],
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get workflow system metrics"""
        total_workflows = len(self.active_workflows) + len(self.workflow_history)
        completed_workflows = len(self.workflow_history)

        avg_quality = 0
        if self.workflow_history:
            quality_scores = [
                w.metrics.get("quality_score", 0) for w in self.workflow_history
            ]
            avg_quality = sum(quality_scores) / len(quality_scores)

        return {
            "total_workflows": total_workflows,
            "active_workflows": len(self.active_workflows),
            "completed_workflows": completed_workflows,
            "average_quality_score": avg_quality,
            "total_templates": len(self.templates),
            "agent_utilization": len(self.agent_assignments),
        }


class WorkflowTemplateManager:
    """Manage workflow templates"""

    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._create_default_templates()

    def _create_default_templates(self):
        """Create default workflow templates"""

        # Literature Review Template
        literature_template = WorkflowTemplate(
            name="Literature Review",
            description="Comprehensive literature review workflow",
            category="research",
            step_templates=[
                {
                    "name": "Topic Definition",
                    "task_type": "coordination",
                    "description": "Define research scope and keywords",
                },
                {
                    "name": "Initial Search",
                    "task_type": "information_gathering",
                    "description": "Conduct initial literature search",
                },
                {
                    "name": "Source Evaluation",
                    "task_type": "evaluation",
                    "description": "Evaluate source quality and relevance",
                },
                {
                    "name": "Content Analysis",
                    "task_type": "data_analysis",
                    "description": "Analyze and extract key insights",
                },
                {
                    "name": "Synthesis",
                    "task_type": "synthesis",
                    "description": "Synthesize findings into coherent review",
                },
                {
                    "name": "Report Generation",
                    "task_type": "reporting",
                    "description": "Generate final literature review report",
                },
            ],
            tags=["literature", "review", "research"],
        )

        self.templates[literature_template.id] = literature_template

        # Comparative Analysis Template
        comparison_template = WorkflowTemplate(
            name="Comparative Analysis",
            description="Compare multiple approaches or solutions",
            category="analysis",
            step_templates=[
                {
                    "name": "Criteria Definition",
                    "task_type": "coordination",
                    "description": "Define comparison criteria",
                },
                {
                    "name": "Data Collection",
                    "task_type": "information_gathering",
                    "description": "Collect data on each option",
                },
                {
                    "name": "Individual Analysis",
                    "task_type": "data_analysis",
                    "description": "Analyze each option separately",
                },
                {
                    "name": "Comparative Evaluation",
                    "task_type": "evaluation",
                    "description": "Compare options against criteria",
                },
                {
                    "name": "Recommendation",
                    "task_type": "synthesis",
                    "description": "Generate recommendations",
                },
            ],
            tags=["comparison", "analysis", "evaluation"],
        )

        self.templates[comparison_template.id] = comparison_template

    def create_workflow_from_template(
        self, template_id: str, name: str, objectives: List[str]
    ) -> Optional[ResearchWorkflow]:
        """Create workflow from template"""
        if template_id not in self.templates:
            return None

        template = self.templates[template_id]
        workflow = ResearchWorkflow(
            name=name,
            description=f"Workflow based on {template.name} template",
            objectives=objectives,
        )

        # Create steps from template
        for i, step_template in enumerate(template.step_templates):
            step = WorkflowStep(
                name=step_template["name"],
                description=step_template["description"],
                task_type=TaskType(step_template["task_type"]),
                dependencies=[workflow.steps[i - 1].id] if i > 0 else [],
            )
            workflow.steps.append(step)

        return workflow

    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates"""
        return [
            {
                "id": template.id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "steps_count": len(template.step_templates),
                "tags": template.tags,
            }
            for template in self.templates.values()
        ]


async def demonstrate_workflow_system():
    """Demonstrate the workflow management system"""
    print("=== Research Workflow Management System Demo ===")

    # Create workflow engine and template manager
    engine = WorkflowEngine()
    template_manager = WorkflowTemplateManager()

    print("\nAvailable Templates:")
    templates = template_manager.list_templates()
    for template in templates:
        print(f"  - {template['name']}: {template['description']}")

    # Create workflow from template
    print("\nCreating workflow from Literature Review template...")
    workflow = template_manager.create_workflow_from_template(
        templates[0]["id"],  # Literature Review template
        "AI in Healthcare Literature Review",
        [
            "Identify current AI applications in healthcare",
            "Analyze effectiveness and limitations",
            "Identify research gaps",
            "Provide recommendations for future research",
        ],
    )

    if workflow:
        # Add to engine
        engine.active_workflows[workflow.id] = workflow
        print(f"Created workflow: {workflow.name}")

        # Plan workflow
        available_agents = ["researcher_001", "analyst_001", "synthesizer_001"]
        success = await engine.plan_workflow(workflow.id, available_agents)
        print(f"Workflow planning: {'Success' if success else 'Failed'}")

        # Optimize workflow
        agent_capabilities = {
            "researcher_001": ["information_gathering", "evaluation"],
            "analyst_001": ["data_analysis", "evaluation"],
            "synthesizer_001": ["synthesis", "reporting"],
        }

        success = await engine.optimize_workflow(workflow.id, agent_capabilities)
        print(f"Workflow optimization: {'Success' if success else 'Failed'}")

        # Start workflow
        success = await engine.start_workflow(workflow.id)
        print(f"Workflow started: {'Success' if success else 'Failed'}")

        # Simulate step completion
        print("\nSimulating workflow execution...")
        for i, step in enumerate(workflow.steps[:3]):  # Complete first 3 steps
            await asyncio.sleep(1)  # Simulate work

            results = {
                "status": "completed",
                "findings": f"Results from {step.name}",
                "confidence": 0.8 + (i * 0.05),
            }

            await engine.complete_step(workflow.id, step.id, results)
            print(f"  Completed step: {step.name}")

        # Get workflow status
        status = engine.get_workflow_status(workflow.id)
        if status:
            print(f"\nWorkflow Status:")
            print(
                f"  Progress: {status['progress']} ({status['progress_percent']:.1f}%)"
            )
            print(f"  Status: {status['status']}")

        # Demonstrate adaptation
        print("\nAdapting workflow based on intermediate results...")
        await engine.adapt_workflow(
            workflow.id, "New insights suggest focusing on specific AI applications"
        )

        # Get system metrics
        metrics = engine.get_system_metrics()
        print(f"\nSystem Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    print("\nWorkflow management demo completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_workflow_system())
