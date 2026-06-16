from vero.core.task.task import VeroTask, TaskT


def create_task(
    name: str,
    register: bool = True,
    task_parameters: type | None = None,
    required_env_vars: list[str] | None = None,
    label_fields: list[str] | None = None,
) -> VeroTask:
    """Create a VeroTask for use in user code.

    Args:
        name: Task name for registry lookup.
        register: Whether to register in the global registry.
        task_parameters: Optional TaskParameters subclass for early validation.
        required_env_vars: Environment variables that must be set for this task
            to run (e.g. ``["LITELLM_BASE_URL", "LITELLM_API_KEY"]``).
        label_fields: Dataset columns that hold labels/ground truth. These are
            stripped from each sample before it is passed to inference, so the
            (agent-authored) inference code never sees them; scoring still gets
            the full row. A static, immutable property of the task definition.

    Returns:
        A new VeroTask instance.
    """
    return VeroTask(
        name=name,
        register=register,
        task_parameters_type=task_parameters,
        required_env_vars=required_env_vars,
        label_fields=label_fields,
    )


__all__ = [
    "VeroTask",
    "TaskT",
    "create_task",
]
