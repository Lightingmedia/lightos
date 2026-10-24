"""LightOS: a cost-optimized multi-cloud job orchestrator.

    from lightos import CostCatalog, Orchestrator, Task

    task = Task.from_yaml("task_config.yaml")
    result = Orchestrator().launch(task)
"""

from .catalog import CatalogError, CostCatalog, estimate_cost
from .execution import (ExecutionError, KubernetesJobExecutor,
                        RemoteJobExecutor)
from .orchestrator import JobResult, Orchestrator
from .task import Task, TaskError

__version__ = "0.1.0"

__all__ = [
    "CostCatalog",
    "CatalogError",
    "estimate_cost",
    "RemoteJobExecutor",
    "KubernetesJobExecutor",
    "ExecutionError",
    "Orchestrator",
    "JobResult",
    "Task",
    "TaskError",
    "__version__",
]
