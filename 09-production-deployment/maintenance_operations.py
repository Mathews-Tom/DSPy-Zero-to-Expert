#!/usr/bin/env python3
"""
Maintenance and Operations System for DSPy Applications

This module provides comprehensive maintenance, operations, and lifecycle management
tools for DSPy applications in production environments.

Learning Objectives:
- Implement automated maintenance workflows
- Create system update and deployment management
- Build health monitoring and diagnostics
- Manage system lifecycle and rollback procedures
- Implement backup and recovery operations
- Handle maintenance scheduling and automation

Author: DSPy Learning Framework
"""

import asyncio
import hashlib
import json
import logging
import shutil
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp
import psutil
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaintenanceType(Enum):
    """Types of maintenance operations"""

    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    ADAPTIVE = "adaptive"


class MaintenanceStatus(Enum):
    """Maintenance operation status"""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLBACK = "rollback"


class DeploymentStrategy(Enum):
    """Deployment strategies"""

    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"


class HealthStatus(Enum):
    """System health status"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MaintenanceWindow:
    """Maintenance window configuration"""

    name: str
    start_time: str  # HH:MM format
    end_time: str  # HH:MM format
    days_of_week: List[str]  # ['monday', 'tuesday', ...]
    timezone: str = "UTC"
    max_duration_hours: int = 4
    allow_emergency: bool = True
    notification_hours: int = 24


@dataclass
class MaintenanceTask:
    """Individual maintenance task"""

    task_id: str
    name: str
    description: str
    maintenance_type: MaintenanceType
    priority: int  # 1-10, 10 being highest
    estimated_duration_minutes: int
    prerequisites: List[str] = field(default_factory=list)
    rollback_procedure: Optional[str] = None
    validation_checks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaintenanceOperation:
    """Complete maintenance operation"""

    operation_id: str
    name: str
    description: str
    maintenance_type: MaintenanceType
    status: MaintenanceStatus
    scheduled_time: datetime
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    tasks: List[MaintenanceTask] = field(default_factory=list)
    maintenance_window: Optional[MaintenanceWindow] = None
    rollback_plan: Optional[str] = None
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    approval_required: bool = True
    approved_by: Optional[str] = None
    executed_by: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """System health information"""

    overall_status: HealthStatus
    timestamp: datetime
    components: Dict[str, HealthStatus] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class BackupInfo:
    """Backup information"""

    backup_id: str
    backup_type: str  # full, incremental, differential
    created_time: datetime
    size_bytes: int
    location: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retention_days: int = 30


class HealthMonitor:
    """Monitor system health and diagnostics"""

    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: deque = deque(maxlen=100)
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.running = False
        self._lock = threading.Lock()

    def register_health_check(
        self, name: str, check_function: Callable, thresholds: Dict[str, float] = None
    ):
        """Register a health check function"""
        self.health_checks[name] = check_function
        if thresholds:
            self.alert_thresholds[name] = thresholds
        logger.info(f"Registered health check: {name}")

    async def run_health_checks(self) -> SystemHealth:
        """Run all health checks and return system health"""
        components = {}
        metrics = {}
        issues = []
        recommendations = []

        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()

                if isinstance(result, dict):
                    components[name] = HealthStatus(result.get("status", "unknown"))
                    if "metrics" in result:
                        metrics.update(
                            {f"{name}_{k}": v for k, v in result["metrics"].items()}
                        )
                    if "issues" in result:
                        issues.extend(result["issues"])
                    if "recommendations" in result:
                        recommendations.extend(result["recommendations"])
                else:
                    components[name] = (
                        HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                    )

            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                components[name] = HealthStatus.UNKNOWN
                issues.append(f"Health check {name} failed: {str(e)}")

        # Determine overall status
        if any(status == HealthStatus.CRITICAL for status in components.values()):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.WARNING for status in components.values()):
            overall_status = HealthStatus.WARNING
        elif any(status == HealthStatus.UNKNOWN for status in components.values()):
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY

        health = SystemHealth(
            overall_status=overall_status,
            timestamp=datetime.utcnow(),
            components=components,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
        )

        with self._lock:
            self.health_history.append(health)

        return health

    async def start_monitoring(self, interval: int = 60):
        """Start continuous health monitoring"""
        self.running = True
        logger.info(f"Starting health monitoring (interval: {interval}s)")

        while self.running:
            try:
                health = await self.run_health_checks()

                # Log critical issues
                if health.overall_status == HealthStatus.CRITICAL:
                    logger.critical(f"System health critical: {health.issues}")
                elif health.overall_status == HealthStatus.WARNING:
                    logger.warning(f"System health warning: {health.issues}")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)

    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        logger.info("Health monitoring stopped")

    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Get health history for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        with self._lock:
            return [h for h in self.health_history if h.timestamp >= cutoff_time]


class BackupManager:
    """Manage system backups and recovery"""

    def __init__(self, backup_root: str = "backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)
        self.backups: Dict[str, BackupInfo] = {}
        self.backup_strategies: Dict[str, Callable] = {}
        self._load_backup_registry()

    def _load_backup_registry(self):
        """Load backup registry from disk"""
        registry_file = self.backup_root / "backup_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    data = json.load(f)
                    for backup_id, backup_data in data.items():
                        backup_data["created_time"] = datetime.fromisoformat(
                            backup_data["created_time"]
                        )
                        self.backups[backup_id] = BackupInfo(**backup_data)
                logger.info(f"Loaded {len(self.backups)} backups from registry")
            except Exception as e:
                logger.error(f"Failed to load backup registry: {e}")

    def _save_backup_registry(self):
        """Save backup registry to disk"""
        registry_file = self.backup_root / "backup_registry.json"
        try:
            data = {}
            for backup_id, backup_info in self.backups.items():
                backup_data = {
                    "backup_id": backup_info.backup_id,
                    "backup_type": backup_info.backup_type,
                    "created_time": backup_info.created_time.isoformat(),
                    "size_bytes": backup_info.size_bytes,
                    "location": backup_info.location,
                    "checksum": backup_info.checksum,
                    "metadata": backup_info.metadata,
                    "retention_days": backup_info.retention_days,
                }
                data[backup_id] = backup_data

            with open(registry_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup registry: {e}")

    def register_backup_strategy(self, name: str, strategy_function: Callable):
        """Register a backup strategy"""
        self.backup_strategies[name] = strategy_function
        logger.info(f"Registered backup strategy: {name}")

    async def create_backup(
        self,
        backup_type: str = "full",
        strategy: str = "default",
        metadata: Dict[str, Any] = None,
    ) -> BackupInfo:
        """Create a system backup"""
        backup_id = f"backup_{int(time.time())}"
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(exist_ok=True)

        logger.info(f"Creating {backup_type} backup: {backup_id}")

        try:
            # Use registered strategy or default
            if strategy in self.backup_strategies:
                await self.backup_strategies[strategy](
                    backup_dir, backup_type, metadata or {}
                )
            else:
                await self._default_backup_strategy(
                    backup_dir, backup_type, metadata or {}
                )

            # Calculate backup size and checksum
            size_bytes = sum(
                f.stat().st_size for f in backup_dir.rglob("*") if f.is_file()
            )
            checksum = self._calculate_backup_checksum(backup_dir)

            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                created_time=datetime.utcnow(),
                size_bytes=size_bytes,
                location=str(backup_dir),
                checksum=checksum,
                metadata=metadata or {},
            )

            self.backups[backup_id] = backup_info
            self._save_backup_registry()

            logger.info(
                f"Backup created successfully: {backup_id} ({size_bytes} bytes)"
            )
            return backup_info

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            # Cleanup failed backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise

    async def _default_backup_strategy(
        self, backup_dir: Path, backup_type: str, metadata: Dict[str, Any]
    ):
        """Default backup strategy"""
        # Backup configuration files
        config_backup = backup_dir / "configs"
        config_backup.mkdir(exist_ok=True)

        # Copy configuration files
        config_sources = ["configs", ".env", "docker-compose.yml", "requirements.txt"]
        for source in config_sources:
            source_path = Path(source)
            if source_path.exists():
                if source_path.is_file():
                    shutil.copy2(source_path, config_backup)
                else:
                    shutil.copytree(
                        source_path,
                        config_backup / source_path.name,
                        dirs_exist_ok=True,
                    )

        # Backup application data
        data_backup = backup_dir / "data"
        data_backup.mkdir(exist_ok=True)

        # Create backup manifest
        manifest = {
            "backup_type": backup_type,
            "created_time": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "contents": {
                "configs": list(config_backup.rglob("*")),
                "data": list(data_backup.rglob("*")),
            },
        }

        with open(backup_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    def _calculate_backup_checksum(self, backup_dir: Path) -> str:
        """Calculate checksum for backup directory"""
        hasher = hashlib.sha256()

        for file_path in sorted(backup_dir.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)

        return hasher.hexdigest()

    async def restore_backup(self, backup_id: str, target_dir: str = None) -> bool:
        """Restore from backup"""
        if backup_id not in self.backups:
            logger.error(f"Backup not found: {backup_id}")
            return False

        backup_info = self.backups[backup_id]
        backup_dir = Path(backup_info.location)

        if not backup_dir.exists():
            logger.error(f"Backup directory not found: {backup_dir}")
            return False

        logger.info(f"Restoring backup: {backup_id}")

        try:
            # Verify backup integrity
            current_checksum = self._calculate_backup_checksum(backup_dir)
            if current_checksum != backup_info.checksum:
                logger.error(f"Backup integrity check failed for {backup_id}")
                return False

            # Restore files
            target_path = Path(target_dir) if target_dir else Path.cwd()

            # Read manifest
            manifest_file = backup_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
                logger.info(f"Restoring backup created at {manifest['created_time']}")

            # Restore configuration files
            config_backup = backup_dir / "configs"
            if config_backup.exists():
                for item in config_backup.rglob("*"):
                    if item.is_file():
                        relative_path = item.relative_to(config_backup)
                        target_file = target_path / relative_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target_file)

            # Restore data files
            data_backup = backup_dir / "data"
            if data_backup.exists():
                for item in data_backup.rglob("*"):
                    if item.is_file():
                        relative_path = item.relative_to(data_backup)
                        target_file = target_path / "data" / relative_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target_file)

            logger.info(f"Backup restored successfully: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False

    def cleanup_old_backups(self):
        """Clean up expired backups"""
        current_time = datetime.utcnow()
        expired_backups = []

        for backup_id, backup_info in self.backups.items():
            expiry_time = backup_info.created_time + timedelta(
                days=backup_info.retention_days
            )
            if current_time > expiry_time:
                expired_backups.append(backup_id)

        for backup_id in expired_backups:
            self.delete_backup(backup_id)

        if expired_backups:
            logger.info(f"Cleaned up {len(expired_backups)} expired backups")

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        if backup_id not in self.backups:
            return False

        backup_info = self.backups[backup_id]
        backup_dir = Path(backup_info.location)

        try:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            del self.backups[backup_id]
            self._save_backup_registry()

            logger.info(f"Deleted backup: {backup_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False

    def list_backups(self) -> List[BackupInfo]:
        """List all available backups"""
        return list(self.backups.values())


class DeploymentManager:
    """Manage application deployments and updates"""

    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.deployment_history: List[Dict[str, Any]] = []
        self.rollback_stack: List[Dict[str, Any]] = []
        self.deployment_strategies: Dict[str, Callable] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default deployment strategies"""
        self.deployment_strategies[DeploymentStrategy.ROLLING.value] = (
            self._rolling_deployment
        )
        self.deployment_strategies[DeploymentStrategy.BLUE_GREEN.value] = (
            self._blue_green_deployment
        )
        self.deployment_strategies[DeploymentStrategy.CANARY.value] = (
            self._canary_deployment
        )
        self.deployment_strategies[DeploymentStrategy.RECREATE.value] = (
            self._recreate_deployment
        )

    async def deploy(
        self,
        version: str,
        strategy: DeploymentStrategy = DeploymentStrategy.ROLLING,
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Deploy a new version of the application"""
        deployment_id = f"deploy_{int(time.time())}"

        logger.info(
            f"Starting deployment {deployment_id}: version {version} using {strategy.value}"
        )

        # Create pre-deployment backup
        backup_info = await self.backup_manager.create_backup(
            backup_type="pre_deployment",
            metadata={"deployment_id": deployment_id, "version": version},
        )

        deployment_record = {
            "deployment_id": deployment_id,
            "version": version,
            "strategy": strategy.value,
            "started_time": datetime.utcnow().isoformat(),
            "backup_id": backup_info.backup_id,
            "config": config or {},
            "status": "in_progress",
        }

        try:
            # Execute deployment strategy
            strategy_func = self.deployment_strategies[strategy.value]
            result = await strategy_func(version, config or {})

            deployment_record.update(
                {
                    "completed_time": datetime.utcnow().isoformat(),
                    "status": "completed",
                    "result": result,
                }
            )

            # Add to rollback stack
            self.rollback_stack.append(deployment_record)

            logger.info(f"Deployment {deployment_id} completed successfully")

        except Exception as e:
            deployment_record.update(
                {
                    "completed_time": datetime.utcnow().isoformat(),
                    "status": "failed",
                    "error": str(e),
                }
            )

            logger.error(f"Deployment {deployment_id} failed: {e}")

            # Attempt automatic rollback
            if self.rollback_stack:
                logger.info("Attempting automatic rollback...")
                await self.rollback()

        finally:
            self.deployment_history.append(deployment_record)

        return deployment_record

    async def _rolling_deployment(
        self, version: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rolling deployment strategy"""
        logger.info(f"Executing rolling deployment to version {version}")

        # Simulate rolling deployment
        nodes = config.get("nodes", ["node1", "node2", "node3"])
        results = {}

        for i, node in enumerate(nodes):
            logger.info(f"Updating node {node} ({i+1}/{len(nodes)})")

            # Simulate node update
            await asyncio.sleep(1)  # Simulate deployment time

            # Health check
            if await self._health_check_node(node):
                results[node] = "success"
                logger.info(f"Node {node} updated successfully")
            else:
                results[node] = "failed"
                raise Exception(f"Health check failed for node {node}")

        return {"strategy": "rolling", "nodes_updated": results}

    async def _blue_green_deployment(
        self, version: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Blue-green deployment strategy"""
        logger.info(f"Executing blue-green deployment to version {version}")

        # Simulate blue-green deployment
        await asyncio.sleep(2)  # Simulate green environment setup

        # Switch traffic
        logger.info("Switching traffic to green environment")
        await asyncio.sleep(1)

        return {"strategy": "blue_green", "environment": "green", "version": version}

    async def _canary_deployment(
        self, version: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Canary deployment strategy"""
        logger.info(f"Executing canary deployment to version {version}")

        canary_percentage = config.get("canary_percentage", 10)

        # Deploy to canary
        logger.info(f"Deploying to {canary_percentage}% of traffic")
        await asyncio.sleep(1)

        # Monitor canary
        logger.info("Monitoring canary deployment...")
        await asyncio.sleep(2)

        # Full rollout
        logger.info("Rolling out to all traffic")
        await asyncio.sleep(1)

        return {
            "strategy": "canary",
            "canary_percentage": canary_percentage,
            "version": version,
        }

    async def _recreate_deployment(
        self, version: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recreate deployment strategy"""
        logger.info(f"Executing recreate deployment to version {version}")

        # Stop old version
        logger.info("Stopping old version")
        await asyncio.sleep(1)

        # Start new version
        logger.info("Starting new version")
        await asyncio.sleep(2)

        return {"strategy": "recreate", "version": version}

    async def _health_check_node(self, node: str) -> bool:
        """Perform health check on a node"""
        # Simulate health check
        await asyncio.sleep(0.5)
        return True  # Assume healthy for demo

    async def rollback(self) -> Dict[str, Any]:
        """Rollback to previous deployment"""
        if not self.rollback_stack:
            raise Exception("No deployments to rollback to")

        current_deployment = self.rollback_stack.pop()
        rollback_id = f"rollback_{int(time.time())}"

        logger.info(f"Rolling back deployment {current_deployment['deployment_id']}")

        try:
            # Restore from backup
            backup_id = current_deployment["backup_id"]
            success = await self.backup_manager.restore_backup(backup_id)

            if not success:
                raise Exception(f"Failed to restore backup {backup_id}")

            rollback_record = {
                "rollback_id": rollback_id,
                "original_deployment_id": current_deployment["deployment_id"],
                "backup_id": backup_id,
                "started_time": datetime.utcnow().isoformat(),
                "completed_time": datetime.utcnow().isoformat(),
                "status": "completed",
            }

            logger.info(f"Rollback {rollback_id} completed successfully")
            return rollback_record

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise


class MaintenanceScheduler:
    """Schedule and manage maintenance operations"""

    def __init__(
        self,
        health_monitor: HealthMonitor,
        backup_manager: BackupManager,
        deployment_manager: DeploymentManager,
    ):
        self.health_monitor = health_monitor
        self.backup_manager = backup_manager
        self.deployment_manager = deployment_manager

        self.maintenance_windows: Dict[str, MaintenanceWindow] = {}
        self.scheduled_operations: Dict[str, MaintenanceOperation] = {}
        self.operation_history: List[MaintenanceOperation] = []
        self.running = False

        # Default maintenance windows
        self._setup_default_windows()

    def _setup_default_windows(self):
        """Set up default maintenance windows"""
        # Weekend maintenance window
        weekend_window = MaintenanceWindow(
            name="weekend_maintenance",
            start_time="02:00",
            end_time="06:00",
            days_of_week=["saturday", "sunday"],
            max_duration_hours=4,
            notification_hours=48,
        )
        self.maintenance_windows["weekend"] = weekend_window

        # Emergency maintenance window (anytime)
        emergency_window = MaintenanceWindow(
            name="emergency_maintenance",
            start_time="00:00",
            end_time="23:59",
            days_of_week=[
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ],
            max_duration_hours=8,
            allow_emergency=True,
            notification_hours=1,
        )
        self.maintenance_windows["emergency"] = emergency_window

    def schedule_maintenance(self, operation: MaintenanceOperation) -> str:
        """Schedule a maintenance operation"""
        operation_id = operation.operation_id

        # Validate maintenance window
        if operation.maintenance_window:
            if not self._is_valid_maintenance_time(
                operation.scheduled_time, operation.maintenance_window
            ):
                raise ValueError(f"Scheduled time is outside maintenance window")

        # Check for conflicts
        conflicts = self._check_scheduling_conflicts(operation)
        if conflicts:
            raise ValueError(f"Scheduling conflicts detected: {conflicts}")

        self.scheduled_operations[operation_id] = operation
        logger.info(f"Scheduled maintenance operation: {operation_id}")

        return operation_id

    def _is_valid_maintenance_time(
        self, scheduled_time: datetime, window: MaintenanceWindow
    ) -> bool:
        """Check if scheduled time is within maintenance window"""
        day_name = scheduled_time.strftime("%A").lower()
        if day_name not in window.days_of_week:
            return False

        scheduled_time_str = scheduled_time.strftime("%H:%M")
        return window.start_time <= scheduled_time_str <= window.end_time

    def _check_scheduling_conflicts(self, operation: MaintenanceOperation) -> List[str]:
        """Check for scheduling conflicts"""
        conflicts = []

        for existing_id, existing_op in self.scheduled_operations.items():
            if existing_op.status in [
                MaintenanceStatus.PLANNED,
                MaintenanceStatus.IN_PROGRESS,
            ]:
                # Check time overlap
                time_diff = abs(
                    (
                        operation.scheduled_time - existing_op.scheduled_time
                    ).total_seconds()
                )
                if time_diff < 3600:  # Less than 1 hour apart
                    conflicts.append(f"Time conflict with operation {existing_id}")

        return conflicts

    async def execute_maintenance(self, operation_id: str) -> MaintenanceOperation:
        """Execute a scheduled maintenance operation"""
        if operation_id not in self.scheduled_operations:
            raise ValueError(f"Operation not found: {operation_id}")

        operation = self.scheduled_operations[operation_id]

        if operation.status != MaintenanceStatus.PLANNED:
            raise ValueError(f"Operation {operation_id} is not in planned status")

        logger.info(f"Executing maintenance operation: {operation_id}")

        # Update status
        operation.status = MaintenanceStatus.IN_PROGRESS
        operation.started_time = datetime.utcnow()

        try:
            # Pre-maintenance health check
            health = await self.health_monitor.run_health_checks()
            if health.overall_status == HealthStatus.CRITICAL:
                raise Exception("System health is critical, aborting maintenance")

            # Create pre-maintenance backup
            backup_info = await self.backup_manager.create_backup(
                backup_type="pre_maintenance",
                metadata={
                    "operation_id": operation_id,
                    "operation_name": operation.name,
                },
            )
            operation.results["backup_id"] = backup_info.backup_id

            # Execute maintenance tasks
            for task in operation.tasks:
                logger.info(f"Executing task: {task.name}")
                await self._execute_maintenance_task(task, operation)

            # Post-maintenance validation
            await self._validate_maintenance(operation)

            # Update status
            operation.status = MaintenanceStatus.COMPLETED
            operation.completed_time = datetime.utcnow()

            logger.info(f"Maintenance operation {operation_id} completed successfully")

        except Exception as e:
            logger.error(f"Maintenance operation {operation_id} failed: {e}")
            operation.status = MaintenanceStatus.FAILED
            operation.completed_time = datetime.utcnow()
            operation.results["error"] = str(e)

            # Attempt rollback if specified
            if operation.rollback_plan:
                await self._execute_rollback(operation)

        finally:
            # Move to history
            self.operation_history.append(operation)
            if operation_id in self.scheduled_operations:
                del self.scheduled_operations[operation_id]

        return operation

    async def _execute_maintenance_task(
        self, task: MaintenanceTask, operation: MaintenanceOperation
    ):
        """Execute a single maintenance task"""
        task_start = time.time()

        try:
            # Check prerequisites
            for prereq in task.prerequisites:
                if not await self._check_prerequisite(prereq):
                    raise Exception(f"Prerequisite not met: {prereq}")

            # Execute task based on type
            if task.maintenance_type == MaintenanceType.SCHEDULED:
                await self._execute_scheduled_task(task)
            elif task.maintenance_type == MaintenanceType.EMERGENCY:
                await self._execute_emergency_task(task)
            elif task.maintenance_type == MaintenanceType.PREVENTIVE:
                await self._execute_preventive_task(task)
            else:
                await self._execute_generic_task(task)

            # Run validation checks
            for check in task.validation_checks:
                if not await self._run_validation_check(check):
                    raise Exception(f"Validation check failed: {check}")

            task_duration = time.time() - task_start
            operation.results[f"task_{task.task_id}"] = {
                "status": "completed",
                "duration_seconds": task_duration,
            }

        except Exception as e:
            task_duration = time.time() - task_start
            operation.results[f"task_{task.task_id}"] = {
                "status": "failed",
                "duration_seconds": task_duration,
                "error": str(e),
            }
            raise

    async def _execute_scheduled_task(self, task: MaintenanceTask):
        """Execute a scheduled maintenance task"""
        logger.info(f"Executing scheduled task: {task.name}")
        # Simulate task execution
        await asyncio.sleep(1)

    async def _execute_emergency_task(self, task: MaintenanceTask):
        """Execute an emergency maintenance task"""
        logger.info(f"Executing emergency task: {task.name}")
        # Simulate urgent task execution
        await asyncio.sleep(0.5)

    async def _execute_preventive_task(self, task: MaintenanceTask):
        """Execute a preventive maintenance task"""
        logger.info(f"Executing preventive task: {task.name}")
        # Simulate preventive maintenance
        await asyncio.sleep(1.5)

    async def _execute_generic_task(self, task: MaintenanceTask):
        """Execute a generic maintenance task"""
        logger.info(f"Executing task: {task.name}")
        # Simulate generic task execution
        await asyncio.sleep(1)

    async def _check_prerequisite(self, prerequisite: str) -> bool:
        """Check if a prerequisite is met"""
        # Simulate prerequisite check
        await asyncio.sleep(0.1)
        return True  # Assume prerequisites are met for demo

    async def _run_validation_check(self, check: str) -> bool:
        """Run a validation check"""
        # Simulate validation check
        await asyncio.sleep(0.2)
        return True  # Assume validation passes for demo

    async def _validate_maintenance(self, operation: MaintenanceOperation):
        """Validate maintenance operation results"""
        logger.info(f"Validating maintenance operation: {operation.operation_id}")

        # Run health checks
        health = await self.health_monitor.run_health_checks()
        operation.results["post_maintenance_health"] = {
            "overall_status": health.overall_status.value,
            "issues": health.issues,
            "recommendations": health.recommendations,
        }

        if health.overall_status == HealthStatus.CRITICAL:
            raise Exception("Post-maintenance health check failed")

    async def _execute_rollback(self, operation: MaintenanceOperation):
        """Execute rollback procedure"""
        logger.info(f"Executing rollback for operation: {operation.operation_id}")

        try:
            operation.status = MaintenanceStatus.ROLLBACK

            # Restore from backup if available
            backup_id = operation.results.get("backup_id")
            if backup_id:
                success = await self.backup_manager.restore_backup(backup_id)
                if success:
                    operation.results["rollback_status"] = "completed"
                    logger.info(
                        f"Rollback completed for operation: {operation.operation_id}"
                    )
                else:
                    operation.results["rollback_status"] = "failed"
                    logger.error(
                        f"Rollback failed for operation: {operation.operation_id}"
                    )

        except Exception as e:
            operation.results["rollback_status"] = "failed"
            operation.results["rollback_error"] = str(e)
            logger.error(f"Rollback execution failed: {e}")

    async def start_scheduler(self):
        """Start the maintenance scheduler"""
        self.running = True
        logger.info("Starting maintenance scheduler")

        while self.running:
            try:
                current_time = datetime.utcnow()

                # Check for scheduled operations
                for operation_id, operation in list(self.scheduled_operations.items()):
                    if (
                        operation.status == MaintenanceStatus.PLANNED
                        and operation.scheduled_time <= current_time
                    ):

                        # Execute maintenance operation
                        await self.execute_maintenance(operation_id)

                # Cleanup old backups
                self.backup_manager.cleanup_old_backups()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    def stop_scheduler(self):
        """Stop the maintenance scheduler"""
        self.running = False
        logger.info("Maintenance scheduler stopped")

    def get_maintenance_status(self) -> Dict[str, Any]:
        """Get current maintenance status"""
        return {
            "scheduled_operations": len(self.scheduled_operations),
            "completed_operations": len(
                [
                    op
                    for op in self.operation_history
                    if op.status == MaintenanceStatus.COMPLETED
                ]
            ),
            "failed_operations": len(
                [
                    op
                    for op in self.operation_history
                    if op.status == MaintenanceStatus.FAILED
                ]
            ),
            "maintenance_windows": list(self.maintenance_windows.keys()),
            "next_scheduled": (
                min([op.scheduled_time for op in self.scheduled_operations.values()])
                if self.scheduled_operations
                else None
            ),
        }


class MaintenanceOperationsManager:
    """Complete maintenance and operations management system"""

    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.backup_manager = BackupManager()
        self.deployment_manager = DeploymentManager(self.backup_manager)
        self.maintenance_scheduler = MaintenanceScheduler(
            self.health_monitor, self.backup_manager, self.deployment_manager
        )

        # Register default health checks
        self._register_default_health_checks()

    def _register_default_health_checks(self):
        """Register default health checks"""

        def cpu_health_check():
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return {
                    "status": "critical",
                    "metrics": {"cpu_percent": cpu_percent},
                    "issues": [f"High CPU usage: {cpu_percent}%"],
                    "recommendations": [
                        "Scale up resources",
                        "Investigate high CPU processes",
                    ],
                }
            elif cpu_percent > 70:
                return {
                    "status": "warning",
                    "metrics": {"cpu_percent": cpu_percent},
                    "issues": [f"Elevated CPU usage: {cpu_percent}%"],
                    "recommendations": ["Monitor CPU usage trends"],
                }
            else:
                return {"status": "healthy", "metrics": {"cpu_percent": cpu_percent}}

        def memory_health_check():
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return {
                    "status": "critical",
                    "metrics": {"memory_percent": memory.percent},
                    "issues": [f"High memory usage: {memory.percent}%"],
                    "recommendations": ["Scale up memory", "Investigate memory leaks"],
                }
            elif memory.percent > 80:
                return {
                    "status": "warning",
                    "metrics": {"memory_percent": memory.percent},
                    "issues": [f"Elevated memory usage: {memory.percent}%"],
                    "recommendations": ["Monitor memory usage trends"],
                }
            else:
                return {
                    "status": "healthy",
                    "metrics": {"memory_percent": memory.percent},
                }

        def disk_health_check():
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                return {
                    "status": "critical",
                    "metrics": {"disk_percent": disk_percent},
                    "issues": [f"High disk usage: {disk_percent:.1f}%"],
                    "recommendations": ["Clean up disk space", "Add more storage"],
                }
            elif disk_percent > 80:
                return {
                    "status": "warning",
                    "metrics": {"disk_percent": disk_percent},
                    "issues": [f"Elevated disk usage: {disk_percent:.1f}%"],
                    "recommendations": ["Monitor disk usage trends"],
                }
            else:
                return {"status": "healthy", "metrics": {"disk_percent": disk_percent}}

        self.health_monitor.register_health_check("cpu", cpu_health_check)
        self.health_monitor.register_health_check("memory", memory_health_check)
        self.health_monitor.register_health_check("disk", disk_health_check)

    async def start_operations(self):
        """Start all maintenance and operations services"""
        logger.info("Starting maintenance and operations system")

        # Start health monitoring
        health_task = asyncio.create_task(self.health_monitor.start_monitoring(60))

        # Start maintenance scheduler
        scheduler_task = asyncio.create_task(
            self.maintenance_scheduler.start_scheduler()
        )

        try:
            await asyncio.gather(health_task, scheduler_task)
        except asyncio.CancelledError:
            logger.info("Operations system stopped")

    def stop_operations(self):
        """Stop all maintenance and operations services"""
        self.health_monitor.stop_monitoring()
        self.maintenance_scheduler.stop_scheduler()
        logger.info("Maintenance and operations system stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "health_monitoring": {
                "running": self.health_monitor.running,
                "registered_checks": len(self.health_monitor.health_checks),
                "history_count": len(self.health_monitor.health_history),
            },
            "backup_management": {
                "total_backups": len(self.backup_manager.backups),
                "backup_strategies": len(self.backup_manager.backup_strategies),
            },
            "deployment_management": {
                "deployment_history": len(self.deployment_manager.deployment_history),
                "rollback_stack": len(self.deployment_manager.rollback_stack),
            },
            "maintenance_scheduling": self.maintenance_scheduler.get_maintenance_status(),
            "timestamp": datetime.utcnow().isoformat(),
        }


async def main():
    """Demonstrate maintenance and operations system"""
    print("=== DSPy Maintenance and Operations System Demo ===")

    # Create operations manager
    ops_manager = MaintenanceOperationsManager()

    print("Starting maintenance and operations system...")
    print("- Health monitoring")
    print("- Backup management")
    print("- Deployment management")
    print("- Maintenance scheduling")

    try:
        # Start operations in background
        ops_task = asyncio.create_task(ops_manager.start_operations())

        # Demonstrate system capabilities
        print("\nDemonstrating system capabilities...")

        # 1. Health monitoring
        print("\n1. Running health checks...")
        health = await ops_manager.health_monitor.run_health_checks()
        print(f"Overall health: {health.overall_status.value}")
        print(f"Components: {list(health.components.keys())}")

        # 2. Backup creation
        print("\n2. Creating system backup...")
        backup_info = await ops_manager.backup_manager.create_backup(
            backup_type="demo",
            metadata={"demo": True, "timestamp": datetime.utcnow().isoformat()},
        )
        print(f"Backup created: {backup_info.backup_id}")
        print(f"Backup size: {backup_info.size_bytes} bytes")

        # 3. Schedule maintenance
        print("\n3. Scheduling maintenance operation...")
        maintenance_task = MaintenanceTask(
            task_id="demo_task_1",
            name="Demo System Update",
            description="Demonstrate maintenance task execution",
            maintenance_type=MaintenanceType.SCHEDULED,
            priority=5,
            estimated_duration_minutes=10,
            validation_checks=["health_check", "connectivity_check"],
        )

        maintenance_op = MaintenanceOperation(
            operation_id="demo_maintenance_1",
            name="Demo Maintenance",
            description="Demonstration of maintenance system",
            maintenance_type=MaintenanceType.SCHEDULED,
            status=MaintenanceStatus.PLANNED,
            scheduled_time=datetime.utcnow() + timedelta(seconds=30),
            tasks=[maintenance_task],
            maintenance_window=ops_manager.maintenance_scheduler.maintenance_windows[
                "weekend"
            ],
        )

        # Override maintenance window for demo
        maintenance_op.maintenance_window = None

        ops_manager.maintenance_scheduler.schedule_maintenance(maintenance_op)
        print(f"Maintenance scheduled: {maintenance_op.operation_id}")

        # 4. Demonstrate deployment
        print("\n4. Demonstrating deployment...")
        deployment_result = await ops_manager.deployment_manager.deploy(
            version="v1.2.0",
            strategy=DeploymentStrategy.ROLLING,
            config={"nodes": ["demo-node-1", "demo-node-2"]},
        )
        print(f"Deployment status: {deployment_result['status']}")

        # Wait for maintenance to execute
        print("\n5. Waiting for scheduled maintenance...")
        await asyncio.sleep(35)

        # Check system status
        print("\n6. System status summary:")
        status = ops_manager.get_system_status()
        print(f"Health checks: {status['health_monitoring']['registered_checks']}")
        print(f"Total backups: {status['backup_management']['total_backups']}")
        print(f"Deployments: {status['deployment_management']['deployment_history']}")
        print(
            f"Scheduled operations: {status['maintenance_scheduling']['scheduled_operations']}"
        )
        print(
            f"Completed operations: {status['maintenance_scheduling']['completed_operations']}"
        )

        print("\nMaintenance and operations demo completed!")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        ops_manager.stop_operations()


if __name__ == "__main__":
    asyncio.run(main())
