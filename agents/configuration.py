from pathlib import Path
from typing import Self

import yaml
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.store.base import BaseStore
from langgraph.config import get_store

from .enums import SearchAPI

NAMESPACE = ("default",)
STORE_CONFIG_KEY = "config"

class Configuration(BaseModel):
    max_web_research_loops: int
    local_llm: str
    search_api: SearchAPI
    prompts: dict[str, str]

    @classmethod
    def from_dict(cls, config_dict: dict) -> Self:
        config_fields = {k: v for k, v in config_dict.items() if k in cls.model_fields}
        return cls(**config_fields)

    @classmethod
    def load(cls, path: str = "config.yml") -> Self:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {path}")
        return cls.from_dict(yaml.safe_load(config_path.read_text()))
    
    @classmethod
    def from_store(cls) -> Self:
        store = get_store()

        try:
            return cls.from_dict(store.get(NAMESPACE, STORE_CONFIG_KEY))
        except AttributeError:
            return cls.load_and_put_in_store(store)
        
    @classmethod
    def load_and_put_in_store(cls, store: BaseStore, *args, **kwargs) -> Self:
        obj = cls.load(*args, **kwargs)
        store.put(NAMESPACE, STORE_CONFIG_KEY, obj.model_dump())
        return obj

    @classmethod
    def from_runnable_config(cls, runnable_config: RunnableConfig) -> Self:
        return cls.from_dict(runnable_config["configurable"])
            
