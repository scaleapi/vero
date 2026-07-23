from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from vero.tools.utils import is_tool


def think(thought: str) -> str:
    """Think and reason about a thought."""
    return ""


class TodoStatus(StrEnum):
    """The status of a todo item."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"


@dataclass
class TodoList:
    """Maintain and keep track of a list of todos."""

    exclude_tools: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    todo_statuses: dict[str, TodoStatus] = field(default_factory=dict)

    @is_tool
    def add_todos(self, tasks: str | list[str]) -> str:
        """Add a todo item or severalitems to the list of todos.

        Args:
            tasks: The task or tasks to add to the list.
        Returns:
            The id of the todo item or items.
        """
        if isinstance(tasks, str):
            tasks = [tasks]

        task_ids = []
        for task in tasks:
            task_ids.append(len(self.todos))
            self.todos.append(task)
            self.todo_statuses[task] = TodoStatus.NOT_STARTED
        return f"{len(tasks)} items have been added to the todo list with ids {', '.join([str(task_id) for task_id in task_ids])}."

    @is_tool
    def update_todo_status(
        self, status: TodoStatus, task: str | None = None, task_id: int | None = None
    ) -> str:
        """Update the status of a todo item. Either provide the task or the task_id.

        Args:
            status: The status to update the todo item to.
            task: The task to update the status of.
            task_id: The id of the todo item to update the status of.
        Returns:
            A message indicating the status of the todo item has been updated.
        """

        if task is None and task_id is None:
            raise ValueError("Either task or task_id must be provided")

        if task_id is not None:
            task = self.todos[task_id]

        self.todo_statuses[task] = status
        return f"Todo item has been updated to status {status}"

    @is_tool
    def list_todos(self, status: list[TodoStatus] | None = None) -> str:
        """List the todos with the given status. Defaults to not started and in progress todos.

        Args:
            status: The statuses to list the todos for.

        Returns:
            A JSON string mapping task ids to task details and their statuses.
        """

        if status is None:
            status = [TodoStatus.NOT_STARTED, TodoStatus.IN_PROGRESS]

        if not isinstance(status, list):
            status = [status]

        todos = {}
        for task_id, task in enumerate(self.todos):
            if self.todo_statuses[task] in status:
                todos[task_id] = {"task": task, "status": self.todo_statuses[task]}

        return f"""Here are the todos with statuses: {status}\n```json{json.dumps(todos, indent=2)}```"""
