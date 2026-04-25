import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_module_imports():
    """Verify all public functions exist in token_buffer module."""
    from food_cooker.memory.token_buffer import create_memory, save_memory, load_memory
    assert callable(create_memory)
    assert callable(save_memory)
    assert callable(load_memory)


def test_settings_has_memory_config():
    from food_cooker import settings
    assert hasattr(settings, "max_token_buffer")
    assert settings.max_token_buffer == 2000