import os
from typing import Optional, Dict, Any, List
from omegaconf import OmegaConf
from openai import AsyncOpenAI as OpenAI

def _build_extra_body(structured_schema: Optional[Dict[str, Any]] = None,
                      backend: str = "lm-format-enforcer") -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if structured_schema is not None:
        extra["guided_json"] = structured_schema
        extra["guided_decoding_backend"] = backend
    return extra

class AsyncChatLLM:
    """
    Wrapper for an (Async) Chat Model.
    """
    def __init__(
        self, 
        kwargs: Dict[str, str],         
        ):
        """
        Initializes AsynceOpenAI client.
        """
        self.model = kwargs.pop("model")
        if self.model == "gpt-4o" or self.model == "gpt-4o" or self.model == "gpt-3.5-turbo-1106":
            pass
        else:
            #OmegaConf.set_struct(kwargs, False) 
            base_url = kwargs.pop("base_url")
            port = kwargs.pop("port")
            version = kwargs.pop("version")
            kwargs["base_url"] = f"{base_url}:{port}/{version}"            
            #OmegaConf.set_struct(kwargs, True)
        
        self.client = OpenAI(**kwargs)

    @property
    async def __call__(self,
        messages: List[Dict[str, str]],
        *,
        structured_schema: Optional[Dict[str, Any]] = None,
        guided_backend: str = "lm-format-enforcer",
        **kwargs,
    ):
        
        """
        Make an async API call.
        """        
        # Mixtral has to follow a different format: ['system', 'assistant', 'user', ...]
        if self.model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            user_message = messages.pop()
            assistant_message = messages.pop()
            assistant_message["role"] = "assistant"
            messages.append(user_message)
            messages.append(assistant_message)
                    
        
        # guided decoding 用の追加パラメータを extra_body で付与（vLLM の拡張）
        extra_body = _build_extra_body(structured_schema, guided_backend)
        if extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}
        
        extra_body = _build_extra_body(structured_schema, guided_backend)
        if extra_body:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra_body}

        return await self.client.chat.completions.create(
            messages=messages,
            **kwargs,
        )
