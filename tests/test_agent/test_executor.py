import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_build_agent_import():
    """Verify build_agent and run_agent are importable without calling them."""
    from food_cooker.agent.executor import build_agent, run_agent
    assert callable(build_agent)
    assert callable(run_agent)


@patch("food_cooker.agent.executor.create_agent")
@patch("food_cooker.agent.executor.get_llm")
@patch("food_cooker.agent.executor.create_memory")
def test_build_agent_calls_components(mock_mem, mock_get_llm, mock_agent_fn):
    from food_cooker.agent.executor import build_agent

    mock_mem.return_value = MagicMock()
    mock_get_llm.return_value = MagicMock()
    mock_agent_fn.return_value = MagicMock()

    agent = build_agent()
    assert agent is not None
    mock_get_llm.assert_called_once()
    mock_agent_fn.assert_called_once()


@patch("food_cooker.agent.executor.create_agent")
@patch("food_cooker.agent.executor.get_llm")
@patch("food_cooker.agent.executor.create_memory")
def test_run_agent_returns_invocable_result(mock_mem, mock_get_llm, mock_agent_fn):
    from food_cooker.agent.executor import run_agent

    mock_mem.return_value = MagicMock()
    mock_agent_instance = MagicMock()
    mock_agent_instance.invoke.return_value = {"input": "test", "output": "test response"}
    mock_agent_fn.return_value = mock_agent_instance

    result = run_agent("test query")
    assert "output" in result
