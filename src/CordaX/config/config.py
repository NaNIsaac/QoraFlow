"""
Global configuration manager with placeholder substitution.
"""
import re
from pathlib import Path
from typing import Any

import yaml

from CordaX.config.definitions import ExpConfig


class ConfigManager:
    # Global State (Singleton role)
    _config_file: Path | str | None = None
    _cached_config: ExpConfig | None = None

    # -------------------- Public API --------------------
    @classmethod
    def initialize(cls, config_file: Path | str) -> None:
        """
        Initialize the ConfigManager with a configuration file path.

        Args:
            config_file (str | Path): Path to the configuration file.
        """
        cls._config_file = config_file
        cls._cached_config = None

    @classmethod
    def load_config(cls, reload: bool = False) -> ExpConfig:
        """
        Load the configuration, optionally reloading.

        Args:
            reload (bool): Whether to force reload the config file.

        Returns:
            ExpConfig: Configuration data object with placeholders resolved.
        """
        if cls._config_file is None:
            raise RuntimeError("Config file not initialized. Call initialize() first.")

        if reload or cls._cached_config is None:
            cls._cached_config = cls.__load_config()

        return cls._cached_config

    @classmethod
    def save_config(cls, config_dict: dict) -> None:
        """
        Save the configuration dictionary back to the file.

        Args:
            config_dict (dict): Configuration dictionary.

        Raises:
            RuntimeError: If config file is not initialized.
        """
        if cls._config_file is None:
            raise RuntimeError("Config file not initialized. Call initialize() first.")

        with open(cls._config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # 캐시 초기화
        cls._cached_config = None

    @classmethod
    def reload(cls) -> ExpConfig:
        """
        Force reload the configuration.
        """
        return cls.load_config(reload=True)

    # -------------------- Internal Helpers --------------------
    @classmethod
    def __load_config(cls) -> ExpConfig:
        with open(cls._config_file, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        context = config_dict.copy()
        replaced_data = cls.replace_placeholders(config_dict, context)

        return ExpConfig(**replaced_data)

    @staticmethod
    def replace_placeholders(data: Any, context: dict, max_iterations=10):
        """
        Recursively replace placeholders in the given data structure.

        Placeholders are in the format `${key.subkey}`.

        Args:
            data (dict, list, str, or any): Data structure containing placeholders.
            context (dict): Context dictionary for placeholder resolution.
            max_iterations (int): Maximum number of iterations to prevent infinite loops.

        Returns:
            Data structure with placeholders replaced.
        """

        def _replace_in_string(s: str, ctx: dict) -> str:
            for _ in range(max_iterations):
                new_s = re.sub(
                    r"\$\{([^}]+)\}",
                    lambda m: str(ConfigManager.resolve_placeholder(m.group(1), ctx)),
                    s,
                )
                if new_s == s:
                    break
                s = new_s
            return s

        match data:
            case dict():
                result = {
                    key: ConfigManager.replace_placeholders(value, context, max_iterations)
                    for key, value in data.items()
                }
                context.update({k: v for k, v in result.items() if isinstance(v, dict | str | int | float | bool)})
                return result
            case list():
                return [ConfigManager.replace_placeholders(item, context, max_iterations) for item in data]
            case str():
                match = re.fullmatch(r"\$\{([^}]+)\}", data)
                if match:
                    return ConfigManager.resolve_placeholder(match.group(1), context)
                return _replace_in_string(data, context)
            case None | int() | float() | bool():
                return data
            case _:
                return data

    @staticmethod
    def resolve_placeholder(placeholder: str, context: dict):
        """
        Resolve a placeholder to its corresponding value from the context.

        Args:
            placeholder (str): Placeholder string in the format `${key.subkey}`.
            context (dict): Context dictionary containing the values.

        Returns:
            Value corresponding to the placeholder.

        Raises:
            ValueError: If the placeholder cannot be resolved.
        """
        keys = placeholder.split(".")
        value = context
        for key in keys:
            if not isinstance(value, dict):
                raise ValueError(f"Cannot resolve placeholder '{placeholder}' (stuck at: {key})")
            value = value.get(key)
            if value is None:
                raise ValueError(f"Placeholder '{placeholder}' could not be resolved")
        return value


if __name__ == "__main__":
    # 최초 초기화
    ConfigManager.initialize("config.yaml")

    # 어디서든 바로 사용 가능
    cfg = ConfigManager.load_config()
    print(cfg)
