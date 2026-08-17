"""
VLLMTokenCounter - 使用vLLM原生接口进行Token计数

确保与vLLM prefill侧完全一致的token计数逻辑。

关键修复：
1. 使用add_special_tokens=False（与vLLM一致）
2. 正确处理max_completion_tokens参数
3. 使用parse_chat_messages处理tool_calls
4. 使用safe_apply_chat_template处理tools定义

用法：
    counter = VLLMTokenCounter(
        model_name="zai-org/GLM-5.1",
        max_model_len=8192,
        chat_template="/path/to/chat_template.jinja",
    )
    
    exceeds, prompt_tokens, total = counter.check_exceeds_limit(
        messages=[{"role": "user", "content": "hello"}],
        tools=[...],
        max_tokens=100,  # 或从请求中获取 max_completion_tokens
    )
"""

import json
import time
from typing import Any, Optional, List, Dict, Tuple

# Lazy import: check vLLM availability only when actually needed
VLLM_AVAILABLE = False
VLLM_IMPORT_ERROR = None

# Use standard logging with vLLM-style format (same as proxy)
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def _check_vllm_available():
    """Check if vLLM is available (lazy check)"""
    global VLLM_AVAILABLE, VLLM_IMPORT_ERROR
    
    if VLLM_AVAILABLE:
        return True
    
    if VLLM_IMPORT_ERROR is not None:
        raise ImportError(f"vLLM not available: {VLLM_IMPORT_ERROR}")
    
    # Try importing vLLM in a specific order to avoid circular imports
    try:
        # Step 1: Import entrypoints modules first (they don't trigger renderers)
        from vllm.entrypoints.chat_utils import parse_chat_messages
        
        # Step 2: Import config (may trigger logger but not renderers)
        from vllm.config import ModelConfig, VllmConfig
        
        # Step 3: Import tokenizers (independent)
        from vllm.tokenizers import cached_get_tokenizer
        from vllm.tokenizers.hf import CachedHfTokenizer
        
        # Step 4: Import renderers.hf LAST (after all dependencies are loaded)
        from vllm.renderers.hf import safe_apply_chat_template
        
        VLLM_AVAILABLE = True
        return True
    except ImportError as e:
        VLLM_IMPORT_ERROR = str(e)
        raise ImportError(f"vLLM not available: {e}")


class VLLMTokenCounter:
    """
    使用vLLM原生接口进行Token计数，确保与vLLM prefill侧100%一致。
    
    核心原理：
    1. 使用vLLM的parse_chat_messages处理tool_calls
    2. 使用safe_apply_chat_template处理chat template和tools
    3. 使用add_special_tokens=False（与vLLM一致）
    4. 正确获取max_completion_tokens或max_tokens
    """
    
    def __init__(
        self,
        model_name: str,
        max_model_len: int,
        chat_template: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
# Check vLLM availability at initialization time
        _check_vllm_available()
        
        # Now import vLLM modules (after check)
        from vllm.config import ModelConfig, VllmConfig
        from vllm.tokenizers.hf import CachedHfTokenizer
        
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.chat_template_path = chat_template
        
        start = time.perf_counter()
        
        # 创建最小化的ModelConfig
        self.model_config = ModelConfig(
            model=model_name,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            tokenizer=model_name,
        )
        
        # 创建最小化的VllmConfig
        self.vllm_config = VllmConfig(model_config=self.model_config)
        
        # 加载tokenizer（支持HuggingFace Hub和本地路径）
        # CachedHfTokenizer.from_pretrained 接受：
        #   - HuggingFace Hub repo id: "model-name"
        #   - 本地目录: "/path/to/model"
        #   - 相对路径: "./local_model"
        self.tokenizer = CachedHfTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        
        # 加载自定义chat template
        if chat_template:
            try:
                with open(chat_template, 'r', encoding='utf-8') as f:
                    self.custom_chat_template = f.read()
                logger.info(f"Loaded custom chat template from: {chat_template}")
            except Exception as e:
                logger.warning(f"Failed to load chat template: {e}")
                self.custom_chat_template = None
        else:
            self.custom_chat_template = None
        
        elapsed = time.perf_counter() - start
        logger.info(
            f"VLLMTokenCounter initialized in {elapsed:.1f}ms: "
            f"model={model_name}, max_len={max_model_len}"
        )
    
    def get_default_max_tokens(self) -> int:
        """Get default max_tokens from model's generation_config or sampling defaults.
        
        This matches what vLLM uses when the user doesn't specify max_tokens:
        model_config.get_diff_sampling_param() → max_new_tokens from generation_config.
        If not available, falls back to max_model_len.
        """
        try:
            diff_params = self.model_config.get_diff_sampling_param()
            default_max = diff_params.get("max_tokens", self.max_model_len)
            logger.info(f"get_default_max_tokens: {default_max} (from model_config.get_diff_sampling_param)")
            return default_max
        except Exception as e:
            logger.warning(f"get_default_max_tokens failed: {e}, fallback to max_model_len={self.max_model_len}")
            return self.max_model_len
    
    def get_override_max_tokens(self) -> Optional[int]:
        """Get override_max_tokens matching vLLM's serving layer logic.
        
        In vLLM, override_max_tokens is computed as:
          - default_sampling_params.get("max_tokens") if generation_config != "auto"/"vllm"
          - else override_generation_config.get("max_new_tokens")
        
        This value is used as a cap in get_max_tokens():
          min(model_max_tokens, fallback_max_tokens, override_max_tokens, platform_max_tokens)
        Returns None if not applicable.
        """
        try:
            mc = self.model_config
            default_sampling_params = mc.get_diff_sampling_param()
            if mc.generation_config not in ("auto", "vllm"):
                override = default_sampling_params.get("max_tokens")
            else:
                override = getattr(mc, "override_generation_config", {}).get("max_new_tokens")
            logger.info(f"get_override_max_tokens: {override} (generation_config={mc.generation_config})")
            return override
        except Exception as e:
            logger.warning(f"get_override_max_tokens failed: {e}")
            return None
    
    def count_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        result = self.analyze_request(messages, tools)
        return result['prompt_tokens']
    
    def count_prompt_tokens(self, prompt: str | list) -> int:
        """Count tokens for completion-style prompt (string or list of strings)."""
        if isinstance(prompt, str):
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))
        elif isinstance(prompt, list):
            return sum(len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompt if isinstance(p, str))
        else:
            logger.warning(f"Unknown prompt type for count_prompt_tokens: {type(prompt)}")
            return 0
    
    def analyze_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """
        分析请求，返回详细的token breakdown。
        
        返回：
          {
            'prompt_tokens': int,      # 总prompt token数（精确）
            'system_tokens': int,      # system prompt token数（近似）
            'tool_tokens': int,        # tool_calls + tools定义 token数（近似）
            'content_tokens': int,     # user/assistant content token数（近似）
          }
        
        prompt_tokens来自完整chat template渲染+tokenize，与vLLM P节点100%一致。
        breakdown字段仅用于日志展示，使用轻量估算而非逐条encode。
        """
        # Import vLLM modules when needed
        from vllm.entrypoints.chat_utils import parse_chat_messages
        from vllm.renderers.hf import safe_apply_chat_template
        
        import copy
        
        try:
            # Step 0: Normalize messages for parse_chat_messages compatibility
            # OpenAI API spec requires tool_calls[].function.arguments to be
            # a JSON string, but some clients send dict directly. Deep copy
            # so we don't modify the caller's original data.
            normalized_messages = copy.deepcopy(messages)
            for msg in normalized_messages:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg.get("tool_calls", []):
                        func = tc.get("function")
                        if isinstance(func, dict):
                            args = func.get("arguments")
                            if isinstance(args, dict):
                                func["arguments"] = json.dumps(args, ensure_ascii=False)
                            elif args is None:
                                func["arguments"] = "{}"
            
            # Step 1: 解析chat messages（处理tool_calls）
            try:
                conversation, mm_data, mm_uuids = parse_chat_messages(
                    normalized_messages,
                    self.model_config,
                    content_format="string",
                )
            except TypeError:
                conversation, mm_data, mm_uuids = parse_chat_messages(
                    normalized_messages,
                    self.model_config,
                )
            
            # Step 2: 应用chat template（处理tools）
            prompt_raw = safe_apply_chat_template(
                self.model_config,
                self.tokenizer,
                conversation,
                tools=tools,
                chat_template=self.custom_chat_template,
                tokenize=False,
            )
            
            if isinstance(prompt_raw, list):
                prompt_raw = "".join(prompt_raw)
            
            # Step 3: tokenize（关键：add_special_tokens=False）
            prompt_token_count = len(self.tokenizer.encode(
                prompt_raw,
                add_special_tokens=False,
            ))
            
            # Step 4: 轻量breakdown估算（不逐条encode，用字符数近似）
            system_tokens = 0
            tool_tokens = 0
            content_tokens = 0
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    text_len = 0
                    if isinstance(content, str):
                        text_len = len(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text_len += len(item["text"])
                    system_tokens += max(text_len // 4, 0)
                
                elif role in ("user", "assistant"):
                    text_len = 0
                    if isinstance(content, str) and content:
                        text_len = len(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text_len += len(item["text"])
                    content_tokens += max(text_len // 4, 0)
                    
                    if "tool_calls" in msg:
                        for tc in msg.get("tool_calls", []):
                            func = tc.get("function", {})
                            args_json = json.dumps(func, ensure_ascii=False) if func else ""
                            tool_tokens += max(len(args_json) // 4, 0)
                
                elif role == "tool":
                    if isinstance(content, str):
                        tool_tokens += max(len(content) // 4, 0)
            
            if tools:
                for tool in tools:
                    if isinstance(tool, dict):
                        func = tool.get("function", {})
                        if isinstance(func, dict):
                            tool_json = json.dumps(func, ensure_ascii=False)
                            tool_tokens += max(len(tool_json) // 4, 0)
            
            return {
                'prompt_tokens': prompt_token_count,
                'system_tokens': system_tokens,
                'tool_tokens': tool_tokens,
                'content_tokens': content_tokens,
            }
            
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, fallback to simple estimation")
            return self._fallback_count(messages, tools)
    
    def _fallback_count(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """简单fallback估算"""
        total = 0
        
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self.tokenizer.encode(content, add_special_tokens=False))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        total += len(self.tokenizer.encode(item["text"], add_special_tokens=False))
            
            if "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    if "name" in func:
                        total += len(self.tokenizer.encode(func["name"], add_special_tokens=False))
                    if "arguments" in func:
                        args = func["arguments"]
                        if isinstance(args, dict):
                            args_json = json.dumps(args, ensure_ascii=False)
                        else:
                            args_json = str(args)
                        total += len(self.tokenizer.encode(args_json, add_special_tokens=False))
        
        if tools:
            for tool in tools:
                func = tool.get("function", tool)
                tool_json = json.dumps(func, ensure_ascii=False)
                total += len(self.tokenizer.encode(tool_json, add_special_tokens=False))
        
        total += 100  # chat template开销
        
        return total
    
    def check_exceeds_limit(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[bool, int, int]:
        """
        检查请求是否超过模型长度限制。
        
        关键修复：正确获取max_completion_tokens或max_tokens
        """
        prompt_tokens = self.count_tokens(messages, tools)
        
        # 关键修复：默认值应为0（如果不指定则vLLM自动计算）
        output_tokens = max_tokens if max_tokens is not None else 0
        total_tokens = prompt_tokens + output_tokens
        
        exceeds = total_tokens > self.max_model_len
        
        return exceeds, prompt_tokens, total_tokens
    
    def get_load_balance_score(self, prompt_tokens: int) -> float:
        """转换为负载均衡score"""
        return prompt_tokens * 0.0345 + 120.0745


# ============================================================================
# 集成说明
# ============================================================================

"""
修改 load_balance_proxy_server_enhanced.py：

1. 导入（约第70行后）：

    try:
        from vllm_token_counter import VLLMTokenCounter
        VLLM_TOKEN_COUNTER_AVAILABLE = True
    except ImportError:
        VLLM_TOKEN_COUNTER_AVAILABLE = False

2. ProxyState.__init__ 增加 vllm_token_counter 字段（约第990行）

3. lifespan() 初始化（约第1775行）：

    vllm_token_counter = None
    if global_args.model_name and VLLM_TOKEN_COUNTER_AVAILABLE:
        try:
            vllm_token_counter = VLLMTokenCounter(
                model_name=global_args.model_name,
                max_model_len=global_args.max_model_len,
                chat_template=global_args.chat_template,
                trust_remote_code=global_args.trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"VLLMTokenCounter failed: {e}")

4. _handle_completions() 使用（约第1463行）：

    # 关键修复：正确获取max_completion_tokens
    max_tokens = req_data.get("max_completion_tokens") or req_data.get("max_tokens", 0)
    
    if proxy_state.vllm_token_counter and messages:
        exceeds, prompt_tokens, total = proxy_state.vllm_token_counter.check_exceeds_limit(
            messages=messages,
            tools=req_data.get("tools"),
            max_tokens=max_tokens,
        )
        if exceeds:
            return Response(status_code=400, ...)
"""
