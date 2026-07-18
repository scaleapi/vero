"""Tests for the tools."""

import json
import re

import pytest
from vero.tools.planning import TodoList, TodoStatus


class TestTodoList:
    """Tests for TodoList tool."""

    def test_add_todos(self):
        """Test adding todo items."""
        todo_list = TodoList()
        todo_list.add_todos("Write tests")
        assert len(todo_list.todos) == 1
        assert todo_list.todo_statuses["Write tests"] == TodoStatus.NOT_STARTED

    def test_update_todo_status(self):
        """Test updating todo status by task_id."""
        todo_list = TodoList()
        todo_list.add_todos("Task 1")
        todo_list.add_todos("Task 2")
        todo_list.update_todo_status(status=TodoStatus.COMPLETED, task_id=0)

        assert len(todo_list.todo_statuses) == 2
        assert todo_list.todo_statuses["Task 1"] == TodoStatus.COMPLETED
        assert todo_list.todo_statuses["Task 2"] == TodoStatus.NOT_STARTED

        result = todo_list.list_todos(status=[TodoStatus.COMPLETED])
        assert "Task 1" in result
        assert "Task 2" not in result

        result = todo_list.list_todos(status=[TodoStatus.NOT_STARTED])
        assert "Task 1" not in result
        assert "Task 2" in result

    def test_update_status_error_handling(self):
        """Test that updating status without task or task_id raises ValueError."""
        todo_list = TodoList()
        todo_list.add_todos("Task 1")

        with pytest.raises(ValueError):
            todo_list.update_todo_status(status=TodoStatus.COMPLETED)

    def test_list_todos(self):
        """Test listing todos with status filters."""
        todo_list = TodoList()
        todo_list.add_todos("Task 1")
        todo_list.add_todos("Task 2")
        todo_list.add_todos("Task 3")

        # Update statuses
        todo_list.update_todo_status(status=TodoStatus.COMPLETED, task_id=0)
        todo_list.update_todo_status(status=TodoStatus.IN_PROGRESS, task_id=1)

        # List completed tasks
        result = todo_list.list_todos(status=TodoStatus.COMPLETED)
        json_match = re.search(r"```json(.+?)```", result, re.DOTALL)
        todos_dict = json.loads(json_match.group(1))
        assert len(todos_dict) == 1
        assert todos_dict["0"]["task"] == "Task 1"
