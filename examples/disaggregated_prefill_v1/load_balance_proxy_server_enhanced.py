# Adapted from load_balance_proxy_server_example.py
# SPDX-License-Identifier: Apache-2.0
#
# Enhanced Load Balance Proxy Server for PD Disaggregation
#
# Based on original load_balance_proxy_server_example.py with ALL original features preserved:
# - Load balancing with priority queues
# - Dynamic add/remove instances with health check
# - NodeListener for waiting nodes and tainted cleanup
# - Taint mechanism with heap priority update
# - Recompute handling for KV transfer
# - Retry logic with exponential backoff
# - Abort request handling
# - Client disconnect cancellation
#
# Enhanced features added on top:
# 1. Prometheus metrics endpoint - vLLM native format output
# 2. Tokenizer-based request validation (early rejection for context limit)
# 3. Backend metrics polling for enhanced load balancing
#
# Prerequisites:
#   pip install fastapi<0.124.0 httpx uvicorn vllm prometheus_client transformers numpy

import argparse
import asyncio
import functools
import hashlib
import heapq
import ipaddress
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# Do NOT import vllm.logger at module level (causes circular import in Python 3.11)
# Use standard logging with vLLM-style format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    datefmt="%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

MAX_RECOMPUTE_RETRIES = 3
_REMOTE_FILL_VERIFICATION_CAPABILITY_BYTES = 32
_DECODER_PLACEMENT_DISCOVERY_TIMEOUT_SECONDS = 2.0
_DECODER_PLACEMENT_POSITIVE_TTL_SECONDS = 30.0
_DECODER_PLACEMENT_NEGATIVE_TTL_SECONDS = 3.0
_BACKEND_CONNECT_TIMEOUT_SECONDS = 10.0
_BACKEND_REQUEST_TIMEOUT_SECONDS = 600.0
_DECODER_READ_TIMEOUT_SECONDS = 120.0
_PREFIX_AFFINITY_HEADER = "x-lmcache-prefix-affinity"
_PREFIX_AFFINITY_MAX_HEADER_BYTES = 512
_PREFIX_AFFINITY_MAX_ANCHORS = 4
_PREFIX_AFFINITY_MAX_KEY_BYTES = 64
_PREFIX_AFFINITY_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PREFIX_AFFINITY_CACHE_SIZE = 65536
_PREFIX_AFFINITY_MIN_TOKENS = 8192
_PREFIX_AFFINITY_MIN_RATIO = 0.25
_PREFIX_AFFINITY_LOAD_SLACK = 1.0

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False
    logger.warning("prometheus_client not installed, /metrics endpoint limited")

try:
    from transformers import AutoTokenizer
    TOKENIZER_ENABLED = True
except ImportError:
    TOKENIZER_ENABLED = False
    logger.warning("transformers not installed, tokenizer validation disabled")

# Add current directory to sys.path for local module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import VLLMTokenCounter (lazy import mechanism, won't fail at import time)
VLLM_TOKEN_COUNTER_AVAILABLE = False
try:
    from vllm_token_counter import VLLMTokenCounter
    VLLM_TOKEN_COUNTER_AVAILABLE = True
    logger.info("VLLMTokenCounter module imported successfully (vLLM will be checked at initialization)")
except ImportError as e:
    # This should NOT happen now due to lazy import
    VLLM_TOKEN_COUNTER_AVAILABLE = False
    logger.warning(f"VLLMTokenCounter import failed unexpectedly: {e}. Fallback to TokenizerAnalyzer.")


# ============================================================================
# Constants (from original)
# ============================================================================

TAINT_PRIORITY = 1e15


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


# ============================================================================
# ServerState (from original + enhanced fields)
# ============================================================================

class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        # Original fields (keep same names as original for priority calculation)
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0
        self.aborted_requests = set()
        # Enhanced fields for backend metrics feedback
        self.backend_running = 0
        self.backend_waiting = 0
        self.backend_kv_usage = 0.0
        self.decoder_remote_fill: dict[int, dict[str, Any]] = {}
        self.decoder_placement_discovered_at = 0.0
        self.decoder_placement_last_attempt_at = 0.0
        self.decoder_rank_active_tokens: dict[int, float] = {}
        self.decoder_placement_task: asyncio.Task[None] | None = None

    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"

    def update_from_backend_metrics(self, metrics: dict):
        """Enhanced: Update state from backend metrics polling"""
        if metrics and "error" not in metrics:
            self.backend_running = metrics.get("num_running", 0)
            self.backend_waiting = metrics.get("num_waiting", 0)
            self.backend_kv_usage = metrics.get("kv_cache_usage", 0.0)


# ============================================================================
# InstanceInfo (from original)
# ============================================================================

@dataclass
class DecoderReservation:
    server: ServerState
    decoder_idx: int
    decoder_score: float
    dp_rank: int | None = None
    api_dp_rank: int | None = None
    preferred_segment: str | None = None
    remote_fill: dict[str, Any] | None = None


@dataclass(frozen=True)
class PrefixAffinityAnchor:
    key: str
    token_end: int


@dataclass(frozen=True)
class PrefixAffinityRecord:
    prefiller_idx: int
    decoder_idx: int
    dp_rank: int
    preferred_segment: str
    destination_engine_epoch: int
    local_tokens: int


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_idx: int
    prefiller_score: float
    prefiller: ServerState
    decoder_idx: int
    decoder_score: float
    decoder: ServerState
    reservation: DecoderReservation | None = None

    def __post_init__(self) -> None:
        if self.reservation is None:
            self.reservation = DecoderReservation(
                self.decoder,
                self.decoder_idx,
                self.decoder_score,
            )


def _parse_prefix_affinity_header(raw_value: str | None) -> tuple[PrefixAffinityAnchor, ...]:
    """Parse bounded ``token_end=opaque_key`` affinity hints."""
    if not raw_value or len(raw_value.encode("utf-8")) > _PREFIX_AFFINITY_MAX_HEADER_BYTES:
        return ()
    items = raw_value.split(",")
    if len(items) > _PREFIX_AFFINITY_MAX_ANCHORS:
        return ()
    anchors: dict[str, int] = {}
    for item in items:
        token_end_text, separator, key = item.partition("=")
        key = key.strip()
        try:
            token_end = int(token_end_text.strip())
        except ValueError:
            return ()
        if (
            not separator
            or token_end <= 0
            or not key
            or len(key.encode("utf-8")) > _PREFIX_AFFINITY_MAX_KEY_BYTES
            or _PREFIX_AFFINITY_KEY_RE.fullmatch(key) is None
        ):
            return ()
        anchors[key] = max(token_end, anchors.get(key, 0))
    ordered = sorted(anchors.items(), key=lambda item: item[1], reverse=True)
    return tuple(PrefixAffinityAnchor(key, token_end) for key, token_end in ordered)


def _automatic_prefix_affinity_anchors(
    request_body: bytes, request_tokens: int
) -> tuple[PrefixAffinityAnchor, ...]:
    """Hash bounded progressive prefixes; affinity is only a placement hint."""
    if not request_body:
        return ()
    body_bytes = len(request_body)
    ends = [end for end in (32 << 10, 64 << 10, 128 << 10, 256 << 10)
            if end < body_bytes]
    if len(ends) < _PREFIX_AFFINITY_MAX_ANCHORS:
        ends.append(body_bytes)
    digest = hashlib.blake2s(digest_size=16)
    anchors = []
    previous = 0
    for end in ends[:_PREFIX_AFFINITY_MAX_ANCHORS]:
        digest.update(request_body[previous:end])
        anchors.append(PrefixAffinityAnchor(
            f"auto-{end:x}-{digest.hexdigest()}",
            max(int(request_tokens) * end // body_bytes, 1),
        ))
        previous = end
    return tuple(reversed(anchors))


# ============================================================================
# RequestAnalysis (Enhanced - complete for Code Agent scenario)
# ============================================================================

@dataclass
class RequestAnalysis:
    """Result of tokenizer analysis - complete breakdown for Code Agent"""
    prompt_tokens: int = 0
    max_tokens: int = 16
    total_tokens: int = 0
    exceeds_limit: bool = False
    exceeded_by: int = 0
    estimated_output_tokens: int = 16
    request_length_bytes: int = 0
    tokenizer_fallback: bool = False
    system_tokens: int = 0
    tool_tokens: int = 0
    content_tokens: int = 0
    analysis_time_ms: float = 0.0


# ============================================================================
# TokenizerAnalyzer (Enhanced Module - complete implementation)
# ============================================================================

class TokenizerAnalyzer:
    """Enhanced: Tokenizer-based request analysis for Code Agent scenario
    
    Supports:
    - Chat template with tools/tool_calls (Code Agent)
    - System prompt, tools, content token breakdown
    - Accurate token counting for complex message formats
    """
    
    def __init__(self, model_name: str, max_model_len: int, trust_remote_code: bool = True,
                 chat_template: Optional[str] = None,
                 default_max_tokens: Optional[int] = None,
                 override_max_tokens: Optional[int] = None,
                 context_length_margin: Optional[int] = None):
        if not TOKENIZER_ENABLED:
            raise ImportError("transformers library not available")
        
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.default_max_tokens = default_max_tokens if default_max_tokens is not None else max_model_len
        self.override_max_tokens = override_max_tokens
        self.context_length_margin = context_length_margin if context_length_margin is not None else 5
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            
            # Load chat template from file path if provided
            if chat_template:
                try:
                    with open(chat_template, 'r', encoding='utf-8') as f:
                        template_content = f.read()
                    self.tokenizer.chat_template = template_content
                    logger.info(f"Loaded custom chat template from: {chat_template}")
                except Exception as e:
                    logger.warning(f"Failed to load chat template from {chat_template}: {e}")
            
            self._vocab_size = len(self.tokenizer)
            logger.info(f"Tokenizer loaded: {model_name}, vocab_size={self._vocab_size}, max_len={max_model_len}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def quick_estimate_exceeds_limit(self, request_body: dict) -> tuple[bool, int]:
        """Quick conservative estimation to filter obviously long requests
        
        Strategy: Overestimate to avoid slow tokenizer calls on long requests
        - If estimate exceeds limit → quick reject (avoid tokenizer timeout)
        - If estimate passes → call tokenizer for precise count
        """
        estimated_tokens = 0
        
        if "prompt" in request_body:
            prompt = request_body["prompt"]
            if isinstance(prompt, str):
                # Add 20% overhead for chat template special tokens
                estimated_tokens = int(len(prompt) // 4 * 1.2)
            elif isinstance(prompt, list):
                # List of prompts
                text_len = sum(len(p) for p in prompt if isinstance(p, str))
                estimated_tokens = int(text_len // 4 * 1.2)
            else:
                estimated_tokens = 200  # fallback
            
        elif "messages" in request_body:
            messages = request_body["messages"]
            
            # Extract content text length (ignore JSON structure)
            total_content_len = 0
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        total_content_len += len(content)
                    elif isinstance(content, list):
                        # Multimodal: extract text
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                total_content_len += len(item["text"])
            
            # Base content tokens
            estimated_tokens = total_content_len // 4
            
            # Add overhead (conservative):
            # - Chat template adds role tokens, formatting tokens
            # - Each message: ~20 tokens overhead (role + format + special tokens)
            num_messages = len(messages)
            estimated_tokens += num_messages * 20
            
            # System prompt often very long in agent scenarios
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_content = msg.get("content", "")
                    if isinstance(system_content, str):
                        # System prompts can be very long, add extra margin
                        estimated_tokens += len(system_content) // 4
            
            # Tool definitions overhead (significant in Code Agent)
            tools = request_body.get("tools", [])
            if tools:
                # Each tool definition: ~100 tokens (name + desc + parameters schema)
                estimated_tokens += len(tools) * 100
                
                # Previous tool calls in messages also add tokens
                for msg in messages:
                    if isinstance(msg, dict):
                        tool_calls = msg.get("tool_calls", [])
                        if tool_calls:
                            # Each tool call: ~30 tokens (function name + arguments)
                            estimated_tokens += len(tool_calls) * 30
        
        # Add generation limit
        max_tokens = request_body.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = request_body.get("max_tokens")
        user_specified_max_tokens = max_tokens is not None
        
        # Context length validation matching vLLM's _token_len_check:
        # No max_tokens → only check prompt, matching vLLM's max_output_tokens=0
        margin = 1 + self.context_length_margin / 100.0
        if user_specified_max_tokens:
            total_estimated = estimated_tokens + max_tokens
            exceeds = total_estimated * margin > self.max_model_len
        else:
            total_estimated = estimated_tokens
            exceeds = estimated_tokens * margin > self.max_model_len
        return exceeds, int(total_estimated)
    
    async def analyze_request_async(self, request_body: dict) -> RequestAnalysis:
        """Async wrapper for analyze_request to avoid blocking event loop"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_request, request_body)
    
    def analyze_request(self, request_body: dict) -> RequestAnalysis:
        """Analyze request to compute token counts with breakdown (sync version)"""
        result = RequestAnalysis()
        
        try:
            prompt_tokens = self.count_tokens(request_body)
            result.prompt_tokens = prompt_tokens
            
            max_tokens = request_body.get("max_completion_tokens")
            if max_tokens is None:
                max_tokens = request_body.get("max_tokens")
            user_specified_max_tokens = max_tokens is not None
            
            # Context length validation matching vLLM's _token_len_check:
            # When user doesn't specify max_tokens, vLLM uses max_output_tokens=0,
            # only checking prompt ≤ max_model_len. We mirror this behavior.
            margin = 1 + self.context_length_margin / 100.0
            
            if user_specified_max_tokens:
                model_max_tokens = self.max_model_len - prompt_tokens
                effective_max_tokens = model_max_tokens
                if max_tokens < effective_max_tokens:
                    effective_max_tokens = max_tokens
                if self.override_max_tokens is not None and self.override_max_tokens < effective_max_tokens:
                    effective_max_tokens = self.override_max_tokens
                
                result.max_tokens = effective_max_tokens
                result.estimated_output_tokens = effective_max_tokens
                result.total_tokens = prompt_tokens + effective_max_tokens
                
                if result.total_tokens * margin > self.max_model_len:
                    result.exceeds_limit = True
                    result.exceeded_by = int(result.total_tokens * margin) - self.max_model_len
            else:
                result.max_tokens = 0
                result.estimated_output_tokens = 0
                result.total_tokens = prompt_tokens
                
                if prompt_tokens * margin > self.max_model_len:
                    result.exceeds_limit = True
                    result.exceeded_by = int(prompt_tokens * margin) - self.max_model_len
            
            if "messages" in request_body:
                messages = request_body["messages"]
                result.request_length_bytes = len(json.dumps(messages))
            
            # Compute breakdown for messages format
            if "messages" in request_body:
                self._compute_breakdown(request_body["messages"], request_body, result)
            
        except Exception as e:
            result.tokenizer_fallback = True
            logger.debug(f"Tokenizer fallback due to error: {e}")
            
            # Fallback estimation (MUST also check limit)
            if "prompt" in request_body:
                prompt = request_body["prompt"]
                if isinstance(prompt, str):
                    result.prompt_tokens = min(len(prompt) // 4, self.max_model_len)
                elif isinstance(prompt, list):
                    result.prompt_tokens = min(len(str(prompt)) // 4, self.max_model_len)
            
            if "messages" in request_body:
                messages = request_body["messages"]
                result.prompt_tokens = min(len(json.dumps(messages)) // 4, self.max_model_len)
            
            # Recalculate total and check limit in fallback
            max_tokens = request_body.get("max_completion_tokens")
            if max_tokens is None:
                max_tokens = request_body.get("max_tokens")
            user_specified_max_tokens = max_tokens is not None
            
            margin = 1 + self.context_length_margin / 100.0
            
            if user_specified_max_tokens:
                model_max_tokens = self.max_model_len - result.prompt_tokens
                effective_max_tokens = model_max_tokens
                if max_tokens < effective_max_tokens:
                    effective_max_tokens = max_tokens
                if self.override_max_tokens is not None and self.override_max_tokens < effective_max_tokens:
                    effective_max_tokens = self.override_max_tokens
                
                result.max_tokens = effective_max_tokens
                result.estimated_output_tokens = effective_max_tokens
                result.total_tokens = result.prompt_tokens + result.estimated_output_tokens
                
                if result.total_tokens * margin > self.max_model_len:
                    result.exceeds_limit = True
                    result.exceeded_by = int(result.total_tokens * margin) - self.max_model_len
            else:
                result.max_tokens = 0
                result.estimated_output_tokens = 0
                result.total_tokens = result.prompt_tokens
                
                if result.prompt_tokens * margin > self.max_model_len:
                    result.exceeds_limit = True
                    result.exceeded_by = int(result.prompt_tokens * margin) - self.max_model_len
        
        return result
    
    def count_tokens(self, request_body: dict) -> int:
        """Count tokens using tokenizer with apply_chat_template for messages format"""
        
        if "prompt" in request_body:
            prompt = request_body["prompt"]
            return self._count_prompt_tokens(prompt)
        
        elif "messages" in request_body:
            messages = request_body["messages"]
            return self._count_messages_tokens(messages, request_body)
        
        else:
            logger.warning("Request body has neither 'prompt' nor 'messages'")
            return 0
    
    def _count_prompt_tokens(self, prompt) -> int:
        """Count tokens for prompt format"""
        if isinstance(prompt, str):
            return len(self.tokenizer.encode(prompt, add_special_tokens=True))
        elif isinstance(prompt, list):
            if all(isinstance(p, str) for p in prompt):
                return sum(len(self.tokenizer.encode(p, add_special_tokens=True)) for p in prompt)
            elif all(isinstance(p, dict) for p in prompt):
                # Prompt as messages format
                return self._count_messages_tokens(prompt, {})
            else:
                total = 0
                for p in prompt:
                    if isinstance(p, str):
                        total += len(self.tokenizer.encode(p, add_special_tokens=True))
                    elif isinstance(p, dict):
                        total += self._count_messages_tokens([p], {})
                return total
        elif isinstance(prompt, dict):
            return self._count_messages_tokens([prompt], {})
        else:
            logger.warning(f"Unknown prompt type: {type(prompt)}")
            return 0
    
    def _count_messages_tokens(self, messages: list, request_body: dict) -> int:
        """Count tokens for messages format using apply_chat_template
        
        Handles:
        - tools parameter (Code Agent scenario)
        - tool_calls in messages
        - System prompts, user content, assistant responses
        
        Note: Uses add_generation_prompt=True and add_special_tokens=True to match
        vLLM's actual token counting behavior for accurate limit checking.
        """
        try:
            tools = request_body.get("tools", None)
            
            # Use apply_chat_template with tools for accurate token count
            # IMPORTANT: Use same parameters as vLLM to get accurate token count
            if tools:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True  # Match vLLM behavior
                )
            else:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # Match vLLM behavior
                )
            
            # Handle list return (some tokenizers return list of strings)
            if isinstance(text, list):
                text = "".join(text)
            
            # IMPORTANT: add_special_tokens=True to include chat template overhead
            return len(self.tokenizer.encode(text, add_special_tokens=True))
            
        except Exception as e:
            logger.debug(f"apply_chat_template failed: {e}, fallback to manual counting")
            return self._count_messages_manual(messages, request_body)
    
    def _count_messages_manual(self, messages: list, request_body: dict) -> int:
        """Fallback manual token counting for messages
        
        Handles:
        - role tokens (system, user, assistant, tool)
        - content (string or list with text/image)
        - tool_calls (function name, arguments)
        """
        total = 0
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                
                # Count role tokens
                if role:
                    total += len(self.tokenizer.encode(role, add_special_tokens=False))
                
                # Count content tokens
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += len(self.tokenizer.encode(content, add_special_tokens=False))
                elif isinstance(content, list):
                    # Content is list of parts (text, image_url, etc.)
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                total += len(self.tokenizer.encode(item["text"], add_special_tokens=False))
                            # Image tokens - some tokenizers have special handling
                            elif "image_url" in item:
                                # Approximate image token count (varies by model)
                                total += 256  # Common default for vision models
                        elif isinstance(item, str):
                            total += len(self.tokenizer.encode(item, add_special_tokens=False))
                
                # Count tool_calls tokens (Code Agent scenario)
                tool_calls = msg.get("tool_calls", [])
                if tool_calls and isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            function = tool_call.get("function", {})
                            if isinstance(function, dict):
                                # Function name
                                name = function.get("name", "")
                                if name:
                                    total += len(self.tokenizer.encode(name, add_special_tokens=False))
                                # Function arguments (JSON string)
                                arguments = function.get("arguments", "")
                                if arguments and isinstance(arguments, str):
                                    total += len(self.tokenizer.encode(arguments, add_special_tokens=False))
        
        # Add tool definitions tokens if present
        tools = request_body.get("tools", [])
        if tools and isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    # Tool type
                    tool_type = tool.get("type", "")
                    if tool_type:
                        total += len(self.tokenizer.encode(tool_type, add_special_tokens=False))
                    # Function definition
                    function = tool.get("function", {})
                    if isinstance(function, dict):
                        for key in ["name", "description"]:
                            value = function.get(key, "")
                            if value:
                                total += len(self.tokenizer.encode(value, add_special_tokens=False))
                        # Parameters schema
                        parameters = function.get("parameters", {})
                        if parameters:
                            total += len(self.tokenizer.encode(json.dumps(parameters), add_special_tokens=False))
        
        return total
    
    def _compute_breakdown(self, messages: list, request_body: dict, result: RequestAnalysis):
        """Compute breakdown of tokens by type: system, tools, content"""
        system_tokens = 0
        tool_tokens = 0
        content_tokens = 0
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                
                if role == "system":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        system_tokens += len(self.tokenizer.encode(content, add_special_tokens=False))
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                system_tokens += len(self.tokenizer.encode(item["text"], add_special_tokens=False))
                
                elif role in ("user", "assistant"):
                    # Content tokens
                    content = msg.get("content", "")
                    if isinstance(content, str) and content:
                        content_tokens += len(self.tokenizer.encode(content, add_special_tokens=False))
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                content_tokens += len(self.tokenizer.encode(item["text"], add_special_tokens=False))
                    
                    # Tool calls tokens (assistant messages in Code Agent)
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls and isinstance(tool_calls, list):
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                function = tool_call.get("function", {})
                                if isinstance(function, dict):
                                    name = function.get("name", "")
                                    if name:
                                        tool_tokens += len(self.tokenizer.encode(name, add_special_tokens=False))
                                    arguments = function.get("arguments", "")
                                    if arguments and isinstance(arguments, str):
                                        tool_tokens += len(self.tokenizer.encode(arguments, add_special_tokens=False))
                
                elif role == "tool":
                    # Tool response content
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        tool_tokens += len(self.tokenizer.encode(content, add_special_tokens=False))
        
        # Tool definitions
        tools = request_body.get("tools", [])
        if tools and isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    function = tool.get("function", {})
                    if isinstance(function, dict):
                        for key in ["name", "description"]:
                            value = function.get(key, "")
                            if value:
                                tool_tokens += len(self.tokenizer.encode(value, add_special_tokens=False))
                        parameters = function.get("parameters", {})
                        if parameters:
                            tool_tokens += len(self.tokenizer.encode(json.dumps(parameters), add_special_tokens=False))
        
        result.system_tokens = system_tokens
        result.tool_tokens = tool_tokens
        result.content_tokens = content_tokens


# ============================================================================
# VLLM Metrics Constants (Enhanced)
# ============================================================================

VLLM_GAUGE_BOTH = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"]

VLLM_COUNTER_PREFILL = [
    "vllm:prompt_tokens",
    "vllm:prompt_tokens_cached",
    "vllm:prompt_tokens_recomputed",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
    "vllm:mm_cache_queries",
    "vllm:mm_cache_hits",
]
VLLM_COUNTER_PREFILL_LABELED = ["vllm:prompt_tokens_by_source"]
VLLM_COUNTER_DECODE = ["vllm:generation_tokens"]
VLLM_COUNTER_DECODE_LABELED = ["vllm:request_success"]
VLLM_COUNTER_BOTH_MERGE = ["vllm:num_preemptions", "vllm:corrupted_requests"]
VLLM_HISTOGRAM_PREFILL = ["vllm:time_to_first_token_seconds", "vllm:request_prefill_time_seconds",
                           "vllm:request_prefill_kv_computed_tokens", "vllm:request_prompt_tokens"]
VLLM_HISTOGRAM_DECODE = ["vllm:inter_token_latency_seconds", "vllm:request_time_per_output_token_seconds",
                          "vllm:request_decode_time_seconds", "vllm:request_generation_tokens"]
VLLM_HISTOGRAM_BOTH = ["vllm:request_queue_time_seconds", "vllm:e2e_request_latency_seconds"]
VLLM_HISTOGRAM_KV_CACHE = ["vllm:kv_block_lifetime_seconds", "vllm:kv_block_idle_before_evict_seconds",
                            "vllm:kv_block_reuse_gap_seconds"]


# ============================================================================
# MetricsAggregator (Enhanced Module - isolated)
# ============================================================================

class MetricsAggregator:
    """Enhanced: Fetch and aggregate metrics from backend nodes"""
    
    def __init__(self, prefillers: list, decoders: list, model_name: str, 
                 poll_interval: float = 5.0, enable_polling: bool = True):
        self.prefillers = prefillers
        self.decoders = decoders
        self.model_name = model_name
        self.poll_interval = poll_interval
        self.enable_polling = enable_polling
        self.prefill_metrics: list[dict] = []
        self.decode_metrics: list[dict] = []
        self.running = False
        self._poll_thread = None
    
    def start(self):
        if not self.enable_polling:
            logger.info("MetricsAggregator polling disabled, API fetch still available")
            return
        if self.running:
            return
        self.running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(f"MetricsAggregator polling started, interval={self.poll_interval}s")
    
    def stop(self):
        self.running = False
    
    def _poll_loop(self):
        while self.running:
            try:
                self._poll_all_nodes_sync()
            except Exception as e:
                logger.error(f"Metrics poll error: {e}")
            time.sleep(self.poll_interval)
    
    def _poll_all_nodes_sync(self):
        for i, s in enumerate(self.prefillers):
            try:
                metrics = self._poll_node_sync(s)
                s.update_from_backend_metrics(metrics)
            except Exception as e:
                logger.debug(f"Poll prefiller {s.host}:{s.port} failed: {e}")
        
        for i, s in enumerate(self.decoders):
            try:
                metrics = self._poll_node_sync(s)
                s.update_from_backend_metrics(metrics)
            except Exception as e:
                logger.debug(f"Poll decoder {s.host}:{s.port} failed: {e}")
    
    def _poll_node_sync(self, server: ServerState) -> dict:
        host = server.host
        if ":" in host and not host.startswith("["):
            try:
                ip = ipaddress.ip_address(host)
                if isinstance(ip, ipaddress.IPv6Address):
                    host = f"[{host}]"
            except Exception:
                pass
        metrics_url = f"http://{host}:{server.port}/metrics"
        
        result = {}
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(metrics_url)
                response.raise_for_status()
                self._parse_lightweight(response.text, result)
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _parse_lightweight(self, text: str, result: dict):
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r'^([\w:]+)(?:\{[^}]*\})?\s+([\d.eE+-]+|NaN)', line)
            if not match:
                continue
            name = match.group(1)
            value_str = match.group(2)
            try:
                value = float(value_str) if value_str != "NaN" else 0.0
            except ValueError:
                continue
            
            if name == "vllm:num_requests_running":
                result["num_running"] = int(value)
            elif name == "vllm:num_requests_waiting":
                result["num_waiting"] = int(value)
            elif name == "vllm:kv_cache_usage_perc":
                result["kv_cache_usage"] = value
    
    async def fetch_all_metrics(self) -> dict:
        """Fetch from all nodes for /metrics endpoint"""
        tasks = []
        for i, s in enumerate(self.prefillers):
            tasks.append(self._fetch_node_async(s, i, "prefill"))
        for i, s in enumerate(self.decoders):
            tasks.append(self._fetch_node_async(s, i, "decode"))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.prefill_metrics = []
        self.decode_metrics = []
        
        for result in results:
            if isinstance(result, dict) and result.get("node_id"):
                if result.get("node_type") == "prefill":
                    self.prefill_metrics.append(result)
                    idx = result.get("idx", -1)
                    if idx >= 0 and idx < len(self.prefillers):
                        self.prefillers[idx].update_from_backend_metrics(result)
                else:
                    self.decode_metrics.append(result)
                    idx = result.get("idx", -1)
                    if idx >= 0 and idx < len(self.decoders):
                        self.decoders[idx].update_from_backend_metrics(result)
        
        return {"prefill": self.prefill_metrics, "decode": self.decode_metrics}
    
    async def _fetch_node_async(self, server: ServerState, idx: int, node_type: str) -> dict:
        host = server.host
        if ":" in host and not host.startswith("["):
            try:
                ip = ipaddress.ip_address(host)
                if isinstance(ip, ipaddress.IPv6Address):
                    host = f"[{host}]"
            except Exception:
                pass
        metrics_url = f"http://{host}:{server.port}/metrics"
        
        result = {"node_id": f"{server.host}:{server.port}", "node_type": node_type, "idx": idx}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(metrics_url)
                response.raise_for_status()
                self._parse_full_metrics(response.text, result)
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _parse_full_metrics(self, text: str, result: dict):
        gauges = {}
        counters = {}
        histograms = {}
        labeled_counters = {}
        
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            match = re.match(r'^([\w:]+)(?:\{([^}]*)\})?\s+([\d.eE+-]+|NaN)', line)
            if not match:
                continue
            
            name = match.group(1)
            labels_str = match.group(2) or ""
            value_str = match.group(3)
            
            try:
                # Parse as float first (handles scientific notation)
                float_value = float(value_str) if value_str != "NaN" else 0.0
            except ValueError:
                continue
            
            # For counters and histogram counts, convert to int (Python handles big ints)
            # Gauge values remain as float
            if name.endswith("_bucket"):
                base_name = name.replace("_bucket", "")
                le_match = re.search(r'le="([^"]+)"', labels_str)
                if le_match:
                    le_value = le_match.group(1)
                    if base_name not in histograms:
                        histograms[base_name] = {"buckets": {}, "sum": 0.0, "count": 0}
                    if le_value != "+Inf":
                        histograms[base_name]["buckets"][float(le_value)] = int(float_value)
                    else:
                        histograms[base_name]["count"] = int(float_value)
            elif name.endswith("_sum"):
                base_name = name.replace("_sum", "")
                if base_name not in histograms:
                    histograms[base_name] = {"buckets": {}, "sum": 0.0, "count": 0}
                histograms[base_name]["sum"] = float_value  # Keep as float for sum
            elif name.endswith("_count") and not name.startswith("vllm:request"):
                base_name = name.replace("_count", "")
                if base_name not in histograms:
                    histograms[base_name] = {"buckets": {}, "sum": 0.0, "count": 0}
                histograms[base_name]["count"] = int(float_value)
            elif name.endswith("_created"):
                continue
            elif name.endswith("_total"):
                base_name = name[:-6]
                if base_name in VLLM_COUNTER_PREFILL_LABELED or base_name in VLLM_COUNTER_DECODE_LABELED:
                    if base_name not in labeled_counters:
                        labeled_counters[base_name] = {}
                    label_key = self._extract_label_key(labels_str)
                    labeled_counters[base_name][label_key] = int(float_value)
                elif base_name in VLLM_COUNTER_PREFILL + VLLM_COUNTER_DECODE + VLLM_COUNTER_BOTH_MERGE:
                    counters[base_name] = int(float_value)  # Python int has arbitrary precision
            elif name in VLLM_GAUGE_BOTH:
                if name not in gauges:
                    gauges[name] = []
                gauges[name].append(float_value)  # Keep as float for gauges
        
        result["gauges"] = gauges
        result["counters"] = counters
        result["histograms"] = histograms
        result["labeled_counters"] = labeled_counters
    
    def _extract_label_key(self, labels_str: str) -> str:
        for part in labels_str.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                if key.strip() in ("source", "finished_reason"):
                    return value.strip('"')
        return labels_str
    
    def generate_vllm_metrics_output(self) -> str:
        lines = []
        lines.append(f"# vLLM Metrics Aggregated from PD Disaggregation")
        lines.append(f"# model_name: {self.model_name}")
        lines.append("")
        
        # Gauge - both prefill/decode
        for gauge_name in VLLM_GAUGE_BOTH:
            prefill_values = []
            decode_values = []
            for m in self.prefill_metrics:
                if gauge_name in m.get("gauges", {}):
                    prefill_values.extend(m["gauges"][gauge_name])
            for m in self.decode_metrics:
                if gauge_name in m.get("gauges", {}):
                    decode_values.extend(m["gauges"][gauge_name])
            
            if prefill_values or decode_values:
                lines.append(f"# HELP {gauge_name} Gauge metric")
                lines.append(f"# TYPE {gauge_name} gauge")
                if prefill_values:
                    agg = sum(prefill_values) / len(prefill_values) if gauge_name == "vllm:kv_cache_usage_perc" else sum(prefill_values)
                    lines.append(f'{gauge_name}{{model_name="{self.model_name}", engine="0-prefill"}} {agg}')
                if decode_values:
                    agg = sum(decode_values) / len(decode_values) if gauge_name == "vllm:kv_cache_usage_perc" else sum(decode_values)
                    lines.append(f'{gauge_name}{{model_name="{self.model_name}", engine="0-decode"}} {agg}')
                lines.append("")
        
        # Counter - prefill only
        for counter_name in VLLM_COUNTER_PREFILL:
            total = sum(m.get("counters", {}).get(counter_name, 0) for m in self.prefill_metrics)
            if total > 0:
                full_name = f"{counter_name}_total"
                lines.append(f"# HELP {full_name} Counter metric")
                lines.append(f"# TYPE {full_name} counter")
                lines.append(f'{full_name}{{model_name="{self.model_name}", engine="0"}} {total}')
                lines.append("")
        
        # Counter - decode only
        for counter_name in VLLM_COUNTER_DECODE:
            total = sum(m.get("counters", {}).get(counter_name, 0) for m in self.decode_metrics)
            if total > 0:
                full_name = f"{counter_name}_total"
                lines.append(f"# HELP {full_name} Counter metric")
                lines.append(f"# TYPE {full_name} counter")
                lines.append(f'{full_name}{{model_name="{self.model_name}", engine="0"}} {total}')
                lines.append("")
        
        # Counter - labeled prefill
        for counter_name in VLLM_COUNTER_PREFILL_LABELED:
            label_totals = {}
            for m in self.prefill_metrics:
                for label_key, value in m.get("labeled_counters", {}).get(counter_name, {}).items():
                    label_totals[label_key] = label_totals.get(label_key, 0) + value
            if label_totals:
                full_name = f"{counter_name}_total"
                lines.append(f"# HELP {full_name} Counter metric")
                lines.append(f"# TYPE {full_name} counter")
                for label_key, total in sorted(label_totals.items()):
                    lines.append(f'{full_name}{{model_name="{self.model_name}", engine="0", source="{label_key}"}} {total}')
                lines.append("")
        
        # Counter - labeled decode
        for counter_name in VLLM_COUNTER_DECODE_LABELED:
            label_totals = {}
            for m in self.decode_metrics:
                for label_key, value in m.get("labeled_counters", {}).get(counter_name, {}).items():
                    label_totals[label_key] = label_totals.get(label_key, 0) + value
            if label_totals:
                full_name = f"{counter_name}_total"
                lines.append(f"# HELP {full_name} Counter metric")
                lines.append(f"# TYPE {full_name} counter")
                for label_key, total in sorted(label_totals.items()):
                    lines.append(f'{full_name}{{model_name="{self.model_name}", engine="0", finished_reason="{label_key}"}} {total}')
                lines.append("")
        
        # Counter - both merge
        for counter_name in VLLM_COUNTER_BOTH_MERGE:
            total = sum(m.get("counters", {}).get(counter_name, 0) for m in self.prefill_metrics + self.decode_metrics)
            if total > 0:
                full_name = f"{counter_name}_total"
                lines.append(f"# HELP {full_name} Counter metric")
                lines.append(f"# TYPE {full_name} counter")
                lines.append(f'{full_name}{{model_name="{self.model_name}", engine="0"}} {total}')
                lines.append("")
        
        # Histogram - prefill only
        for hist_name in VLLM_HISTOGRAM_PREFILL:
            agg = self._aggregate_histograms(self.prefill_metrics, hist_name)
            if agg["count"] > 0:
                lines.extend(self._format_histogram(hist_name, agg, "0"))
                lines.append("")
        
        # Histogram - decode only
        for hist_name in VLLM_HISTOGRAM_DECODE:
            agg = self._aggregate_histograms(self.decode_metrics, hist_name)
            if agg["count"] > 0:
                lines.extend(self._format_histogram(hist_name, agg, "0"))
                lines.append("")
        
        # Histogram - both (two outputs)
        for hist_name in VLLM_HISTOGRAM_BOTH:
            prefill_agg = self._aggregate_histograms(self.prefill_metrics, hist_name)
            decode_agg = self._aggregate_histograms(self.decode_metrics, hist_name)
            if prefill_agg["count"] > 0:
                lines.extend(self._format_histogram(hist_name, prefill_agg, "0-prefill"))
            if decode_agg["count"] > 0:
                lines.extend(self._format_histogram(hist_name, decode_agg, "0-decode"))
            if prefill_agg["count"] > 0 or decode_agg["count"] > 0:
                lines.append("")
        
        # KV Cache histogram - both
        for hist_name in VLLM_HISTOGRAM_KV_CACHE:
            prefill_agg = self._aggregate_histograms(self.prefill_metrics, hist_name)
            decode_agg = self._aggregate_histograms(self.decode_metrics, hist_name)
            if prefill_agg["count"] > 0:
                lines.extend(self._format_histogram(hist_name, prefill_agg, "0-prefill"))
            if decode_agg["count"] > 0:
                lines.extend(self._format_histogram(hist_name, decode_agg, "0-decode"))
            if prefill_agg["count"] > 0 or decode_agg["count"] > 0:
                lines.append("")
        
        return "\n".join(lines)
    
    def _aggregate_histograms(self, metrics_list: list, hist_name: str) -> dict:
        total_sum = 0.0
        total_count = 0
        merged_buckets = {}
        for m in metrics_list:
            hist_data = m.get("histograms", {}).get(hist_name, {})
            total_sum += hist_data.get("sum", 0.0)
            total_count += hist_data.get("count", 0)
            for le, count in hist_data.get("buckets", {}).items():
                merged_buckets[le] = merged_buckets.get(le, 0) + count
        return {"sum": total_sum, "count": total_count, "buckets": merged_buckets}
    
    def _format_histogram(self, name: str, data: dict, engine: str) -> list:
        lines = []
        lines.append(f"# HELP {name} Histogram metric")
        lines.append(f"# TYPE {name} histogram")
        for le, count in sorted(data.get("buckets", {}).items()):
            lines.append(f'{name}_bucket{{model_name="{self.model_name}", engine="{engine}", le="{le}"}} {count}')
        lines.append(f'{name}_bucket{{model_name="{self.model_name}", engine="{engine}", le="+Inf"}} {data.get("count", 0)}')
        lines.append(f'{name}_sum{{model_name="{self.model_name}", engine="{engine}"}} {data.get("sum", 0)}')
        lines.append(f'{name}_count{{model_name="{self.model_name}", engine="{engine}"}} {data.get("count", 0)}')
        return lines


# ============================================================================
# ProxyState (from original + enhanced fields)
# ============================================================================

class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances, 
                 tokenizer_analyzer=None, metrics_aggregator=None, max_model_len=None,
                 vllm_token_counter=None, default_max_tokens=None, override_max_tokens=None,
                 context_length_margin=None, enable_remote_lmcache_store=False,
                 enable_prefix_affinity_routing=False):
        # Original fields
        self.request_num = 0
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []
        self.node_listener = None
        
        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.enable_remote_lmcache_store = bool(enable_remote_lmcache_store)
        self.enable_prefix_affinity_routing = bool(
            enable_prefix_affinity_routing and enable_remote_lmcache_store
        )
        self.prefix_affinity: OrderedDict[str, PrefixAffinityRecord] = OrderedDict()
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        
        # Thread safety: RLock protects heap/list modifications across
        # NodeListener thread and main event loop (reentrant for nested calls)
        self._state_lock = threading.RLock()
        # Timestamp counter for heap tie-breaking (avoid idx bias)
        self._timestamp_counter = 0
        
        # Priority queues: (priority, timestamp, idx, server)
        # timestamp ensures fair selection when priorities are equal
        self.prefiller_heap = [(0, self._next_timestamp(), i, server) for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0, self._next_timestamp(), i, server) for i, server in enumerate(self.decoders)]
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)
        
        # Enhanced fields
        self.tokenizer_analyzer = tokenizer_analyzer
        self.metrics_aggregator = metrics_aggregator
        self.max_model_len = max_model_len
        self.vllm_token_counter = vllm_token_counter
        self.default_max_tokens = default_max_tokens
        self.override_max_tokens = override_max_tokens
        self.context_length_margin = context_length_margin if context_length_margin is not None else 5
    
    def _next_timestamp(self) -> int:
        """Get next timestamp for heap tie-breaking"""
        self._timestamp_counter += 1
        return self._timestamp_counter
    
    # Load balancing methods - based on real token counts
    def _update_prefiller_priority(self, server_idx: int):
        with self._state_lock:
            server = self.prefillers[server_idx]
            priority = server.active_tokens + server.active_kv_cache * 0.3
            self.prefiller_heap = [(p, ts, i, s) for p, ts, i, s in self.prefiller_heap if i != server_idx]
            heapq.heappush(self.prefiller_heap, (priority, self._next_timestamp(), server_idx, server))
    
    def _update_decoder_priority(self, server_idx: int):
        with self._state_lock:
            server = self.decoders[server_idx]
            priority = server.active_tokens
            self.decoder_heap = [(p, ts, i, s) for p, ts, i, s in self.decoder_heap if i != server_idx]
            heapq.heappush(self.decoder_heap, (priority, self._next_timestamp(), server_idx, server))
    
    def abort_prefiller_request(self, server_idx: int, request_id):
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)
    
    def acquire_aborted_prefiller_requests(self, server_idx: int):
        if server_idx >= len(self.prefillers):
            return set()
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests
    
    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())
    
    # Add calculate methods (same as original)
    def calculate_prefill_scores(self, request_length: int) -> float:
        # Check if using original logic
        if global_args.use_original_lb:
            # ORIGINAL: bytes-based formula
            length_score = request_length / 4.0
            input_score = length_score * 0.0345 + 120.0745
            return input_score
        else:
            # ENHANCED: real token-based formula
            length_score = request_length
            input_score = length_score * 0.0345 + 120.0745
            return input_score
    
    def calculate_decode_scores(self, request_length: int) -> float:
        # Check if using original logic
        if global_args.use_original_lb:
            # ORIGINAL: bytes directly
            return request_length
        else:
            # ENHANCED: scaled to avoid huge scores
            return request_length * 0.1
    
    def select_prefiller(self, token_count, preferred_idx: int | None = None):
        """Select prefiller based on score. Returns idx."""
        with self._state_lock:
            if not self.prefiller_heap:
                raise RuntimeError("No prefiller servers available")
            if preferred_idx is not None and not 0 <= preferred_idx < len(self.prefillers):
                raise IndexError("Preferred prefiller index is out of range")
            chosen = preferred_idx
            if chosen is None:
                chosen = heapq.heappop(self.prefiller_heap)[2]
            self.prefillers[chosen].active_tokens += token_count
            self.prefillers[chosen].active_kv_cache += token_count
            self._update_prefiller_priority(chosen)
            return chosen
    
    def release_prefiller(self, idx: int, token_count):
        """Release prefill phase (prefill completed)."""
        with self._state_lock:
            if idx >= len(self.prefillers):
                return
            if self.prefillers[idx].active_tokens >= token_count:
                self.prefillers[idx].active_tokens -= token_count
            elif self.prefillers[idx].active_tokens > 0:
                self.prefillers[idx].active_tokens = 0
            self._update_prefiller_priority(idx)
    
    def release_prefiller_kv(self, idx: int, token_count):
        """Release KV cache (decode started)."""
        with self._state_lock:
            if idx >= len(self.prefillers):
                return
            if self.prefillers[idx].active_kv_cache >= token_count:
                self.prefillers[idx].active_kv_cache -= token_count
            elif self.prefillers[idx].active_kv_cache > 0:
                self.prefillers[idx].active_kv_cache = 0
            self._update_prefiller_priority(idx)
    
    def select_decoder(self, token_count, preferred_idx: int | None = None):
        """Select decoder based on score. Returns idx."""
        with self._state_lock:
            if not self.decoder_heap:
                raise RuntimeError("No decoder servers available")
            if preferred_idx is not None and not 0 <= preferred_idx < len(self.decoders):
                raise IndexError("Preferred decoder index is out of range")
            chosen = preferred_idx
            if chosen is None:
                chosen = heapq.heappop(self.decoder_heap)[2]
            self.decoders[chosen].active_tokens += token_count
            self._update_decoder_priority(chosen)
            return chosen
    
    def release_decoder(self, idx: int, token_count, dp_rank: int | None = None):
        """Release decode phase (decode completed)."""
        with self._state_lock:
            if idx >= len(self.decoders):
                return
            server = self.decoders[idx]
            if dp_rank in server.decoder_rank_active_tokens:
                server.decoder_rank_active_tokens[dp_rank] = max(
                    0.0,
                    server.decoder_rank_active_tokens[dp_rank] - token_count,
                )
                if (
                    server.decoder_rank_active_tokens[dp_rank] == 0
                    and dp_rank not in server.decoder_remote_fill
                ):
                    server.decoder_rank_active_tokens.pop(dp_rank)
            if self.decoders[idx].active_tokens >= token_count:
                self.decoders[idx].active_tokens -= token_count
            elif self.decoders[idx].active_tokens > 0:
                self.decoders[idx].active_tokens = 0
            self._update_decoder_priority(idx)

    def assign_decoder_rank(
        self,
        reservation: DecoderReservation,
        preferred_dp_rank: int | None = None,
    ) -> DecoderReservation:
        with self._state_lock:
            server = reservation.server
            if not server.decoder_remote_fill:
                return reservation
            if preferred_dp_rank not in server.decoder_remote_fill:
                preferred_dp_rank = min(
                    server.decoder_remote_fill,
                    key=lambda rank: (server.decoder_rank_active_tokens[rank], rank),
                )
            dp_rank = preferred_dp_rank
            server.decoder_rank_active_tokens[dp_rank] += reservation.decoder_score
            reservation.dp_rank = dp_rank
            remote_fill = server.decoder_remote_fill.get(dp_rank)
            if remote_fill is not None:
                reservation.remote_fill = dict(remote_fill)
                reservation.api_dp_rank = reservation.remote_fill.pop("api_dp_rank")
                reservation.preferred_segment = reservation.remote_fill.pop(
                    "mooncake_preferred_segment", None
                )
            return reservation

    def _affinity_record_is_current(self, record: PrefixAffinityRecord) -> bool:
        if record.prefiller_idx >= len(self.prefillers) or record.decoder_idx >= len(self.decoders):
            return False
        decoder = self.decoders[record.decoder_idx]
        placement = decoder.decoder_remote_fill.get(record.dp_rank)
        return bool(
            placement
            and record.dp_rank in decoder.decoder_rank_active_tokens
            and placement.get("mooncake_preferred_segment") == record.preferred_segment
            and placement.get("destination_engine_epoch")
            == record.destination_engine_epoch
        )

    def _affinity_is_meaningful(
        self,
        anchor: PrefixAffinityAnchor,
        record: PrefixAffinityRecord,
        request_tokens: int,
    ) -> bool:
        local_tokens = min(anchor.token_end, record.local_tokens)
        return (
            local_tokens >= _PREFIX_AFFINITY_MIN_TOKENS
            and local_tokens / max(request_tokens, 1)
            >= _PREFIX_AFFINITY_MIN_RATIO
        )

    def resolve_prefix_affinity(
        self,
        anchors: tuple[PrefixAffinityAnchor, ...],
        request_tokens: int,
        prefiller_score: float,
        decoder_score: float,
    ) -> tuple[PrefixAffinityAnchor | None, PrefixAffinityRecord | None, bool]:
        """Return the longest known prefix and whether its owner is admissible."""
        if not self.enable_prefix_affinity_routing or not anchors:
            return None, None, False
        request_tokens = max(request_tokens, anchors[0].token_end)
        with self._state_lock:
            for anchor in anchors:
                record = self.prefix_affinity.get(anchor.key)
                if record is None:
                    continue
                if not self._affinity_record_is_current(record):
                    self.prefix_affinity.pop(anchor.key, None)
                    continue
                self.prefix_affinity.move_to_end(anchor.key)
                if not self._affinity_is_meaningful(anchor, record, request_tokens):
                    return anchor, record, False
                prefiller = self.prefillers[record.prefiller_idx]
                decoder = self.decoders[record.decoder_idx]
                rank_best = min(decoder.decoder_rank_active_tokens.values())
                selected = (
                    prefiller not in self.tainted_prefillers
                    and decoder not in self.tainted_decoders
                    and prefiller.active_tokens + prefiller.active_kv_cache * 0.3
                    <= self.prefiller_heap[0][0]
                    + prefiller_score * _PREFIX_AFFINITY_LOAD_SLACK
                    and decoder.active_tokens
                    <= self.decoder_heap[0][0]
                    + decoder_score * _PREFIX_AFFINITY_LOAD_SLACK
                    and decoder.decoder_rank_active_tokens[record.dp_rank]
                    <= rank_best
                    + decoder_score * _PREFIX_AFFINITY_LOAD_SLACK
                )
                return anchor, record, selected
        return None, None, False

    def record_prefix_affinity(
        self,
        anchors: tuple[PrefixAffinityAnchor, ...],
        required_end: int,
        prefiller_idx: int,
        reservation: DecoderReservation,
        local_tokens: int,
    ) -> str | None:
        if (
            not self.enable_prefix_affinity_routing
            or reservation.dp_rank is None
            or reservation.preferred_segment is None
            or reservation.remote_fill is None
        ):
            return None
        valid_anchors = tuple(item for item in anchors if (
            _PREFIX_AFFINITY_MIN_TOKENS <= item.token_end <= required_end
        ))
        if not valid_anchors:
            return None
        epoch = reservation.remote_fill.get("destination_engine_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            return None
        with self._state_lock:
            for anchor in valid_anchors:
                self.prefix_affinity[anchor.key] = PrefixAffinityRecord(
                    prefiller_idx=prefiller_idx,
                    decoder_idx=reservation.decoder_idx,
                    dp_rank=reservation.dp_rank,
                    preferred_segment=reservation.preferred_segment,
                    destination_engine_epoch=epoch,
                    local_tokens=min(max(local_tokens, 0), anchor.token_end),
                )
                self.prefix_affinity.move_to_end(anchor.key)
            while len(self.prefix_affinity) > _PREFIX_AFFINITY_CACHE_SIZE:
                self.prefix_affinity.popitem(last=False)
        return valid_anchors[0].key

    def _decoder_placement_is_fresh(self, server: ServerState, now: float) -> bool:
        if server.decoder_placement_discovered_at:
            ttl = (
                _DECODER_PLACEMENT_POSITIVE_TTL_SECONDS
                if server.decoder_remote_fill
                else _DECODER_PLACEMENT_NEGATIVE_TTL_SECONDS
            )
            if now - server.decoder_placement_discovered_at < ttl:
                return True
        return (
            server.decoder_placement_last_attempt_at > 0
            and now - server.decoder_placement_last_attempt_at
            < _DECODER_PLACEMENT_NEGATIVE_TTL_SECONDS
        )

    async def _refresh_decoder_remote_fill(self, server: ServerState) -> None:
        task = asyncio.current_task()
        try:
            remote_fill = await _discover_decoder_remote_fill(
                server,
                timeout_seconds=_DECODER_PLACEMENT_DISCOVERY_TIMEOUT_SECONDS,
            )
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            logger.warning(
                "Mooncake placement discovery failed for decoder %s: %s",
                server.url,
                exc,
            )
        else:
            with self._state_lock:
                for dp_rank in remote_fill:
                    server.decoder_rank_active_tokens.setdefault(dp_rank, 0.0)
                server.decoder_remote_fill = remote_fill
                for dp_rank, load in list(
                    server.decoder_rank_active_tokens.items()
                ):
                    if dp_rank not in remote_fill and load == 0:
                        server.decoder_rank_active_tokens.pop(dp_rank)
                server.decoder_placement_discovered_at = time.monotonic()
        finally:
            if server.decoder_placement_task is task:
                server.decoder_placement_task = None

    async def ensure_decoder_remote_fill(
        self,
        server: ServerState,
        *,
        wait_for_result: bool = False,
    ) -> None:
        now = time.monotonic()
        if self._decoder_placement_is_fresh(server, now):
            return
        task = server.decoder_placement_task
        if task is None:
            server.decoder_placement_last_attempt_at = now
            task = asyncio.create_task(self._refresh_decoder_remote_fill(server))
            server.decoder_placement_task = task
        if wait_for_result:
            await asyncio.shield(task)

    async def add_instances(self, instance_type: str, instances: list[ServerState]) -> tuple[list[str], list[str]]:
        added_nodes, waiting_nodes = [], []
        for server in instances:
            is_valid = await self.node_listener.check_instance_status(server.client)
            if is_valid and instance_type == InstanceType.PREFILL:
                self.add_prefillers([server])
                added_nodes.append(str(server))
            elif is_valid and instance_type == InstanceType.DECODE:
                self.add_decoders([server])
                added_nodes.append(str(server))
            else:
                node = str(server)
                self.node_listener.waiting_nodes[node] = (instance_type, server, 0)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes
    
    def add_prefillers(self, instances: list[ServerState]) -> None:
        with self._state_lock:
            for server in instances:
                if server in self.tainted_prefillers:
                    self.tainted_prefillers.remove(server)
                    self.prefiller_heap = [
                        (0, self._next_timestamp(), idx, server) if srv == server else (priority, ts, idx, srv)
                        for priority, ts, idx, srv in self.prefiller_heap
                    ]
                    heapq.heapify(self.prefiller_heap)
                elif server not in self.prefillers:
                    self.prefillers.append(server)
                    heapq.heappush(self.prefiller_heap, (0, self._next_timestamp(), len(self.prefillers) - 1, server))
            self.print_status(f"Add prefiller instances: {instances}.")
    
    def add_decoders(self, instances: list[ServerState]) -> None:
        with self._state_lock:
            for server in instances:
                if server in self.tainted_decoders:
                    self.tainted_decoders.remove(server)
                    self.decoder_heap = [
                        (0, self._next_timestamp(), idx, server) if srv == server else (priority, ts, idx, srv)
                        for priority, ts, idx, srv in self.decoder_heap
                    ]
                    heapq.heapify(self.decoder_heap)
                elif server not in self.decoders:
                    self.decoders.append(server)
                    heapq.heappush(self.decoder_heap, (0, self._next_timestamp(), len(self.decoders) - 1, server))
            self.print_status(f"Add decoder instances: {instances}.")
    
    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        with self._state_lock:
            if not instances:
                return False
            if self.request_num > 0:
                logger.warning(f"Start to taint prefill instances {instances}.")
                self._taint_prefillers(instances)
                return True
            instances_to_remove = set(instances)
            self.prefillers = [server for server in self.prefillers if server not in instances_to_remove]
            prefiller_heap_copy = self.prefiller_heap.copy()
            prefiller_heap_copy.sort(key=lambda x: x[2])
            prefiller_heap = []
            idx = 0
            for priority, ts, _, server in prefiller_heap_copy:
                if server not in instances_to_remove:
                    prefiller_heap.append((priority, ts, idx, server))
                    idx += 1
            self.prefiller_heap = prefiller_heap
            heapq.heapify(self.prefiller_heap)
            self.print_status(f"Remove prefiller instances: {instances}.")
            return False
    
    def remove_decoders(self, instances: list[ServerState]) -> bool:
        with self._state_lock:
            if not instances:
                return False
            if self.request_num > 0:
                logger.warning(f"Start to taint decode instances {instances}.")
                self._taint_decoders(instances)
                return True
            instances_to_remove = set(instances)
            self.decoders = [server for server in self.decoders if server not in instances_to_remove]
            decoder_heap_copy = self.decoder_heap.copy()
            decoder_heap_copy.sort(key=lambda x: x[2])
            decoder_heap = []
            idx = 0
            for priority, ts, _, server in decoder_heap_copy:
                if server not in instances_to_remove:
                    decoder_heap.append((priority, ts, idx, server))
                    idx += 1
            self.decoder_heap = decoder_heap
            heapq.heapify(self.decoder_heap)
            self.print_status(f"Remove decoder instances: {instances}.")
            return False
    
    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        with self._state_lock:
            instances_to_taint = set(instances)
            for server in self.prefillers:
                if server in instances_to_taint and server not in self.tainted_prefillers:
                    self.tainted_prefillers.append(server)
            self.prefiller_heap = [
                (TAINT_PRIORITY, ts, idx, srv) if srv in instances_to_taint else (priority, ts, idx, srv)
                for priority, ts, idx, srv in self.prefiller_heap
            ]
            heapq.heapify(self.prefiller_heap)
    
    def _taint_decoders(self, instances: list[ServerState]) -> None:
        with self._state_lock:
            instances_to_taint = set(instances)
            for server in self.decoders:
                if server in instances_to_taint and server not in self.tainted_decoders:
                    self.tainted_decoders.append(server)
            self.decoder_heap = [
                (TAINT_PRIORITY, ts, idx, srv) if srv in instances_to_taint else (priority, ts, idx, srv)
                for priority, ts, idx, srv in self.decoder_heap
            ]
            heapq.heapify(self.decoder_heap)
    
    def print_status(self, msg: str) -> None:
        status = {
            "prefill_instances": [str(server) for server in self.prefillers],
            "decode_instances": [str(server) for server in self.decoders],
        }
        logger.info(f"{msg} Status: {status}")
    
    # Enhanced method
    def get_stats(self) -> dict:
        return {
            "prefill_instances": [str(s) for s in self.prefillers],
            "decode_instances": [str(s) for s in self.decoders],
            "tainted_prefillers": [str(s) for s in self.tainted_prefillers],
            "tainted_decoders": [str(s) for s in self.tainted_decoders],
            "request_num": self.request_num,
            "prefiller_heap_size": len(self.prefiller_heap),
            "decoder_heap_size": len(self.decoder_heap),
            "prefiller_token_load": {
                str(s): {
                    "active_tokens": s.active_tokens,
                    "active_kv_cache": s.active_kv_cache,
                } for s in self.prefillers
            },
            "decoder_token_load": {
                str(s): {
                    "active_tokens": s.active_tokens,
                } for s in self.decoders
            },
            "backend_metrics": {
                "prefillers": [{"host": s.host, "port": s.port, "running": s.backend_running, "waiting": s.backend_waiting} for s in self.prefillers],
                "decoders": [{"host": s.host, "port": s.port, "running": s.backend_running, "waiting": s.backend_waiting} for s in self.decoders],
            }
        }


# ============================================================================
# NodeListener (from original)
# ============================================================================

class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        self.waiting_nodes: dict[str, tuple[str, Any, int]] = {}
        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()
    
    def _node_listener(self) -> None:
        while True:
            for node, (instance_type, server, check_times) in list(self.waiting_nodes.items()):
                is_valid = self._check_instance_status_sync(server)
                logger.info(f"Checking instance {node}...")
                check_times += 1
                if is_valid:
                    if instance_type == InstanceType.PREFILL:
                        self.proxy_state.add_prefillers([server])
                    else:
                        self.proxy_state.add_decoders([server])
                    self.waiting_nodes.pop(node)
                elif check_times == global_args.max_waiting_retries:
                    logger.info(f"Instance {node} was not added to the proxy.")
                    self.waiting_nodes.pop(node)
                else:
                    self.waiting_nodes[node] = (instance_type, server, check_times)
            
            with self.proxy_state._state_lock:
                if self.proxy_state.tainted_prefillers and not self.proxy_state.request_num:
                    need_waiting = self.proxy_state.remove_prefillers(self.proxy_state.tainted_prefillers)
                    if not need_waiting:
                        self.proxy_state.tainted_prefillers.clear()
            
            with self.proxy_state._state_lock:
                if self.proxy_state.tainted_decoders and not self.proxy_state.request_num:
                    need_waiting = self.proxy_state.remove_decoders(self.proxy_state.tainted_decoders)
                    if not need_waiting:
                        self.proxy_state.tainted_decoders.clear()
            
            time.sleep(global_args.waiting_retry_interval)
    
    @staticmethod
    async def check_instance_status(client: httpx.AsyncClient) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False

    @staticmethod
    def _check_instance_status_sync(server: ServerState) -> bool:
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            with httpx.Client(timeout=5.0, base_url=server.url) as client:
                response = client.get("/models", headers=headers)
                response.raise_for_status()
                return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def _parse_decoder_remote_fill_response(payload: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("Decoder collective RPC returned an invalid response")
    placements: dict[int, dict[str, Any]] = {}
    for result in payload["results"]:
        if result is None:
            continue
        if not isinstance(result, dict):
            raise ValueError("Decoder placement result must be a dictionary")
        remote_fill = result.get("remote_fill")
        if remote_fill is None:
            continue
        if not isinstance(remote_fill, dict):
            raise ValueError("Decoder remote-fill placement must be a dictionary")
        if remote_fill.get("enabled") is not True:
            continue
        dp_rank = result.get("dp_rank")
        api_dp_rank = result.get("api_dp_rank")
        advertised_dp_rank = remote_fill.get("dp_rank")
        advertised_tp_rank = remote_fill.get("tp_rank")
        if (
            isinstance(dp_rank, bool)
            or not isinstance(dp_rank, int)
            or dp_rank < 0
            or isinstance(advertised_dp_rank, bool)
            or not isinstance(advertised_dp_rank, int)
            or advertised_dp_rank != dp_rank
            or isinstance(api_dp_rank, bool)
            or not isinstance(api_dp_rank, int)
            or api_dp_rank < 0
            or isinstance(advertised_tp_rank, bool)
            or not isinstance(advertised_tp_rank, int)
            or advertised_tp_rank != 0
        ):
            raise ValueError("Decoder remote-fill placement is not bound to its TP0/DP rank")
        required_strings = (
            "destination_engine_id",
            "control_endpoint",
            "token_hash_algorithm",
            "descriptor_verification_capability",
        )
        if any(
            not isinstance(remote_fill.get(name), str)
            or not remote_fill[name].strip()
            for name in required_strings
        ):
            raise ValueError("Decoder remote-fill string identity is invalid")
        epoch = remote_fill.get("destination_engine_epoch")
        generation = remote_fill.get("shared_cache_generation")
        tp_size = remote_fill.get("destination_tp_size")
        dp_size = remote_fill.get("destination_dp_size")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("Decoder remote-fill engine epoch is invalid")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("Decoder remote-fill shared-cache generation is invalid")
        if (
            isinstance(tp_size, bool)
            or not isinstance(tp_size, int)
            or tp_size <= 0
            or isinstance(dp_size, bool)
            or not isinstance(dp_size, int)
            or dp_size <= 0
            or dp_rank >= dp_size
        ):
            raise ValueError("Decoder remote-fill parallel topology is invalid")
        global_te_push = remote_fill.get("global_te_push")
        if not isinstance(global_te_push, bool):
            raise ValueError("Decoder remote-fill native capability is invalid")
        if not global_te_push:
            continue
        verification_capability = remote_fill["descriptor_verification_capability"]
        try:
            verification_key = bytes.fromhex(verification_capability)
        except ValueError as error:
            raise ValueError("Decoder remote-fill verification capability is invalid") from error
        if (
            len(verification_key) != _REMOTE_FILL_VERIFICATION_CAPABILITY_BYTES
            or verification_key.hex() != verification_capability
        ):
            raise ValueError("Decoder remote-fill verification capability is invalid")
        python_hash_seed = remote_fill.get("python_hash_seed", "")
        if not isinstance(python_hash_seed, str) or (
            remote_fill["token_hash_algorithm"] == "builtin" and not python_hash_seed
        ):
            raise ValueError("Decoder remote-fill hash identity is invalid")
        advertised_segment = remote_fill.get("destination_remote_session")
        segment = result.get("segment")
        if any(
            value is not None
            and (not isinstance(value, str) or not value.strip())
            for value in (advertised_segment, segment)
        ):
            raise ValueError("Decoder Mooncake segment is invalid")
        advertised_segment = (
            advertised_segment.strip() if advertised_segment else None
        )
        segment = segment.strip() if segment else None
        if segment and advertised_segment and segment != advertised_segment:
            raise ValueError("Decoder Mooncake placement identities disagree")
        segment = segment or advertised_segment
        placement = {
            "api_dp_rank": api_dp_rank,
            "destination_engine_id": remote_fill["destination_engine_id"].strip(),
            "destination_engine_epoch": epoch,
            "control_endpoint": remote_fill["control_endpoint"].strip(),
            "destination_dp_rank": dp_rank,
            "shared_cache_generation": generation,
            "destination_tp_size": tp_size,
            "destination_dp_size": dp_size,
            "global_te_push": global_te_push,
            "token_hash_algorithm": remote_fill["token_hash_algorithm"].strip(),
            "python_hash_seed": python_hash_seed,
            "descriptor_verification_capability": verification_capability,
        }
        if segment is not None:
            placement["mooncake_preferred_segment"] = segment
        existing = placements.get(dp_rank)
        if existing is not None and existing != placement:
            raise ValueError(
                f"Decoder DP rank {dp_rank} reported conflicting remote-fill metadata"
            )
        placements[dp_rank] = placement
    return placements


async def _discover_decoder_remote_fill(
    server: ServerState,
    *,
    timeout_seconds: float = 2.0,
) -> dict[int, dict[str, Any]]:
    collective_rpc_url = server.url.removesuffix("/v1") + "/collective_rpc"
    response = await server.client.post(
        collective_rpc_url,
        json={"method": "get_mooncake_placement_info"},
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    return _parse_decoder_remote_fill_response(payload)


# ============================================================================
# Request Handlers (from original)
# ============================================================================

def _http_timeout(read_timeout: float) -> httpx.Timeout:
    return httpx.Timeout(
        connect=_BACKEND_CONNECT_TIMEOUT_SECONDS,
        read=read_timeout,
        write=_BACKEND_CONNECT_TIMEOUT_SECONDS,
        pool=_BACKEND_CONNECT_TIMEOUT_SECONDS,
    )


def _decoder_headers(
    request_id: str,
    decoder_api_dp_rank: int | None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    if decoder_api_dp_rank is not None:
        headers["X-data-parallel-rank"] = str(decoder_api_dp_rank)
    return headers


def _sse_error(code: str, message: str) -> bytes:
    error = {"error": {"message": message, "type": "backend_error", "code": code}}
    payload = json.dumps(error, separators=(",", ":"))
    return f"data: {payload}\n\n".encode()


class _IncompleteDecoderStreamError(RuntimeError):
    pass


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    remote_fill_handoff: dict[str, Any] | None = None,
    preferred_mooncake_segment: str | None = None,
):
    aborted_requests = proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
    }
    if remote_fill_handoff is not None:
        req_data["kv_transfer_params"]["lmcache.remote_fill"] = dict(
            remote_fill_handoff
        )
        if preferred_mooncake_segment is not None:
            req_data["kv_transfer_params"][
                "lmcache.mooncake_preferred_segment"
            ] = preferred_mooncake_segment
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    try:
        response = await asyncio.wait_for(
            client.post(
                endpoint,
                json=req_data,
                headers=headers,
                timeout=_http_timeout(global_args.backend_request_timeout),
            ),
            timeout=global_args.backend_request_timeout,
        )
        response.raise_for_status()
        return response
    except BaseException:
        if prefiller_id < len(proxy_state.prefillers):
            proxy_state.prefillers[prefiller_id].aborted_requests.update(
                aborted_requests
            )
        raise


async def stream_decoder_response(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    decoder_api_dp_rank: int | None = None,
):
    async with client.stream(
        "POST",
        endpoint,
        json=req_data,
        headers=_decoder_headers(request_id, decoder_api_dp_rank),
        timeout=_http_timeout(global_args.decoder_read_timeout),
    ) as response:
        response.raise_for_status()
        sse_buffer = bytearray()
        async for raw_chunk in response.aiter_bytes():
            sse_buffer.extend(raw_chunk)
            while b"\n\n" in sse_buffer:
                event_end = sse_buffer.index(b"\n\n") + 2
                event = bytes(sse_buffer[:event_end])
                del sse_buffer[:event_end]
                logger.debug(f"[SSE_BUFFER] yield event: {event[:300]}")
                yield event
                if event == b"data: [DONE]\n\n":
                    return
        if sse_buffer:
            raise _IncompleteDecoderStreamError(
                "decoder stream ended with an incomplete SSE event"
            )
        raise _IncompleteDecoderStreamError(
            "decoder stream ended without [DONE]"
        )


async def _handle_select_instance(
    api: str,
    req_data: Any,
    request_length: int,
    analysis=None,
    original_request_id=None,
    prefix_anchors: tuple[PrefixAffinityAnchor, ...] = (),
):
    prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    decoder_score = proxy_state.calculate_decode_scores(request_length)
    known_anchor, known_affinity, affinity_selected = (
        proxy_state.resolve_prefix_affinity(
            prefix_anchors,
            request_length,
            prefiller_score,
            decoder_score,
        )
    )
    request_id = await proxy_state.next_req_id()
    prefiller_idx = None
    prefiller_active_released = False
    reservation = None
    learned_affinity_key = None
    try:
        if proxy_state.enable_remote_lmcache_store:
            decoder_idx = proxy_state.select_decoder(
                decoder_score,
                known_affinity.decoder_idx if affinity_selected else None,
            )
            decoder = proxy_state.decoders[decoder_idx]
            reservation = DecoderReservation(decoder, decoder_idx, decoder_score)
            await proxy_state.ensure_decoder_remote_fill(
                decoder,
                wait_for_result=True,
            )
            proxy_state.assign_decoder_rank(
                reservation, known_affinity.dp_rank if affinity_selected else None
            )
            if affinity_selected and (
                reservation.dp_rank != known_affinity.dp_rank
                or reservation.preferred_segment
                != known_affinity.preferred_segment
                or reservation.remote_fill is None
                or reservation.remote_fill.get("destination_engine_epoch")
                != known_affinity.destination_engine_epoch
            ):
                proxy_state.release_decoder(
                    reservation.decoder_idx,
                    reservation.decoder_score,
                    reservation.dp_rank,
                )
                affinity_selected = False
                decoder_idx = proxy_state.select_decoder(decoder_score)
                decoder = proxy_state.decoders[decoder_idx]
                reservation = DecoderReservation(decoder, decoder_idx, decoder_score)
                await proxy_state.ensure_decoder_remote_fill(
                    decoder,
                    wait_for_result=True,
                )
                proxy_state.assign_decoder_rank(reservation)

        prefiller_idx = proxy_state.select_prefiller(
            prefiller_score,
            known_affinity.prefiller_idx if affinity_selected else None,
        )
        prefiller = proxy_state.prefillers[prefiller_idx]
        remote_fill_handoff = None
        if reservation is not None and reservation.remote_fill is not None:
            remote_fill_handoff = {
                **reservation.remote_fill,
                "transfer_id": uuid.uuid4().hex,
                "request_attempt": 1,
                "source_engine_id": str(prefiller),
            }
        response = await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            request_id,
            remote_fill_handoff=remote_fill_handoff,
            preferred_mooncake_segment=(
                reservation.preferred_segment
                if remote_fill_handoff is not None and reservation is not None
                else None
            ),
        )
        proxy_state.release_prefiller(prefiller_idx, prefiller_score)
        prefiller_active_released = True
        response_json = response.json()
        if not isinstance(response_json, dict):
            raise ValueError(
                f"P-node returned invalid JSON for request {request_id}: "
                f"type={type(response_json).__name__}"
            )
        returned_params = response_json.get("kv_transfer_params")
        if returned_params is None:
            returned_params = {}
        if not isinstance(returned_params, dict):
            raise TypeError("Prefiller kv_transfer_params must be a dictionary")
        kv_transfer_params = dict(returned_params)
        returned_remote_fill = kv_transfer_params.pop("lmcache.remote_fill", None)
        kv_transfer_params.pop("lmcache.remote_fill_result", None)
        if remote_fill_handoff is not None:
            if not isinstance(returned_remote_fill, dict):
                raise RuntimeError("Prefiller omitted remote-fill terminal result")
            terminal = returned_remote_fill.get("terminal")
            if not isinstance(terminal, dict):
                raise RuntimeError("Prefiller returned invalid remote-fill terminal result")
            outcome = terminal.get("outcome")
            persistent_end = terminal.get("persistent_common_end")
            required_end = terminal.get("required_store_end")
            if (
                outcome not in {"LOCAL_FULL", "PERSISTENT_ONLY"}
                or isinstance(persistent_end, bool)
                or not isinstance(persistent_end, int)
                or isinstance(required_end, bool)
                or not isinstance(required_end, int)
                or persistent_end < required_end
                or required_end < 0
                or terminal.get("transfer_id") != remote_fill_handoff["transfer_id"]
            ):
                raise RuntimeError(
                    "Prefiller remote-fill result is not safe for decoder forwarding"
                )
            if outcome == "LOCAL_FULL":
                kv_transfer_params["lmcache.remote_fill_result"] = {
                    "outcome": outcome,
                    "required_store_end": required_end,
                    "destination_engine_epoch": remote_fill_handoff[
                        "destination_engine_epoch"
                    ],
                }
            known_is_meaningful = bool(known_anchor and known_affinity and (
                proxy_state._affinity_is_meaningful(
                    known_anchor,
                    known_affinity,
                    max(request_length, prefix_anchors[0].token_end),
                )
            ))
            if affinity_selected and known_anchor and known_affinity:
                local_tokens = known_affinity.local_tokens + max(
                    0, required_end - known_anchor.token_end
                )
            elif known_is_meaningful:
                local_tokens = -1
            else:
                local_tokens = required_end - (known_anchor.token_end if known_anchor else 0)
            if local_tokens >= 0:
                learned_affinity_key = proxy_state.record_prefix_affinity(
                    prefix_anchors,
                    required_end,
                    prefiller_idx,
                    reservation,
                    local_tokens,
                )
        kv_transfer_params.pop("lmcache.mooncake_preferred_segment", None)
        kv_transfer_params.pop("lmcache.mooncake_preferred_kv_group", None)
        req_data["kv_transfer_params"] = kv_transfer_params
        if kv_transfer_params:
            logger.debug(f"[{request_id}] KV transfer params received from P-node: "
                         f"engine_id={kv_transfer_params.get('remote_engine_id')}, "
                         f"block_ids_count={len(kv_transfer_params.get('remote_block_ids') or ())}, "
                         f"host={kv_transfer_params.get('remote_host')}, "
                         f"port={kv_transfer_params.get('remote_port')}")
        else:
            logger.warning(f"[{request_id}] P-node returned EMPTY kv_transfer_params! "
                           f"D-node will not have KV cache location info. "
                           f"P-node response keys: {list(response_json.keys())}, "
                           f"P-node: {prefiller}")

        if reservation is None:
            decoder_idx = proxy_state.select_decoder(decoder_score)
            decoder = proxy_state.decoders[decoder_idx]
            reservation = DecoderReservation(decoder, decoder_idx, decoder_score)

        max_tokens = req_data.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = req_data.get("max_tokens")
        if max_tokens is None:
            max_tokens = "default"
        analysis_time_str = ""
        if analysis and analysis.analysis_time_ms > 0:
            analysis_time_str = f"analysis={analysis.analysis_time_ms:.1f}ms, "

        if analysis:
            token_info = f"prompt={analysis.prompt_tokens}, max_gen={max_tokens}"
            if analysis.system_tokens > 0 or analysis.tool_tokens > 0:
                token_info += f", sys={analysis.system_tokens}, tools={analysis.tool_tokens}, content={analysis.content_tokens}"
            token_info += f", total={analysis.total_tokens}"
        else:
            token_info = f"bytes={request_length}"

        affinity_info = ""
        if affinity_selected and known_anchor:
            affinity_info = f", affinity=local:{known_anchor.token_end}"
        elif known_anchor:
            affinity_info = f", affinity=balanced:{known_anchor.token_end}"
        elif learned_affinity_key:
            affinity_info = ", affinity=learned"
        recompute_info = f" [recompute of {original_request_id}]" if original_request_id else ""
        logger.info(
            "[%s] %s%s%s → P:%s(%.1f) → D:%s(%.1f)%s",
            request_id,
            analysis_time_str,
            token_info,
            affinity_info,
            prefiller,
            prefiller_score,
            decoder,
            decoder_score,
            recompute_info,
        )
        return InstanceInfo(
            request_id=request_id,
            prefiller_idx=prefiller_idx,
            prefiller_score=prefiller_score,
            prefiller=prefiller,
            decoder_idx=reservation.decoder_idx,
            decoder_score=decoder_score,
            decoder=decoder,
            reservation=reservation,
        )
    except BaseException:
        if reservation is not None:
            proxy_state.release_decoder(
                reservation.decoder_idx,
                reservation.decoder_score,
                reservation.dp_rank,
            )
        if prefiller_idx is not None:
            if not prefiller_active_released:
                proxy_state.release_prefiller(prefiller_idx, prefiller_score)
            proxy_state.abort_prefiller_request(prefiller_idx, request_id)
            proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
        raise


def _release_decoder_reservation(instance_info: InstanceInfo) -> None:
    reservation = instance_info.reservation
    if reservation is None:
        return
    instance_info.reservation = None
    proxy_state.release_decoder(
        reservation.decoder_idx,
        reservation.decoder_score,
        reservation.dp_rank,
    )


class _CleanupStreamingResponse(StreamingResponse):
    def __init__(
        self,
        *args: Any,
        cleanup: Callable[[], None],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cleanup()


async def _handle_completions(api: str, request: Request):
    instance_info = None
    response_owns_cleanup = False
    request_count_released = False
    released_kv = True

    def release_request_count() -> None:
        nonlocal request_count_released
        if request_count_released:
            return
        proxy_state.request_num = max(0, proxy_state.request_num - 1)
        request_count_released = True

    def cleanup_current_request() -> None:
        nonlocal released_kv
        try:
            if instance_info is not None and not released_kv:
                proxy_state.abort_prefiller_request(
                    instance_info.prefiller_idx,
                    instance_info.request_id,
                )
                proxy_state.release_prefiller_kv(
                    instance_info.prefiller_idx,
                    instance_info.prefiller_score,
                )
                released_kv = True
        finally:
            try:
                if instance_info is not None:
                    _release_decoder_reservation(instance_info)
            finally:
                release_request_count()

    try:
        proxy_state.request_num += 1
        req_data = await request.json()
        prefix_anchors = _parse_prefix_affinity_header(
            getattr(request, "headers", {}).get(_PREFIX_AFFINITY_HEADER)
        )
        
        # Normalize tool_calls[].function.arguments: dict → JSON string
        # OpenAI API spec requires arguments to be a JSON string, but some
        # clients (e.g. Code Agent frameworks) send dict directly.
        # This normalization ensures both VLLMTokenCounter and P node vLLM
        # can parse the request correctly via Pydantic validation.
        normalized_count = 0
        for msg in req_data.get("messages", []):
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function")
                    if isinstance(func, dict):
                        args = func.get("arguments")
                        if isinstance(args, dict):
                            func["arguments"] = json.dumps(args, ensure_ascii=False)
                            normalized_count += 1
                        elif args is None:
                            func["arguments"] = "{}"
                            normalized_count += 1
        if normalized_count > 0:
            logger.info(f"[NORMALIZE] Converted {normalized_count} tool_calls.arguments from dict to JSON string")
        
        # Check for multimodal content in messages - reject if model doesn't support it
        # Models like GLM-5 are text-only and will fail on vLLM backend if
        # multimodal content (image_url, input_audio, video_url) is included.
        mm_parts_found = False
        mm_part_types = set()
        for msg in req_data.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type in ("image_url", "input_audio", "video_url", "image"):
                            mm_parts_found = True
                            mm_part_types.add(part_type)
                        # Also check for image_url inside the part
                        if "image_url" in part:
                            mm_parts_found = True
                            mm_part_types.add("image_url")
        
        if mm_parts_found:
            release_request_count()
            logger.info(f"[REJECTED] Multimodal content detected: {mm_part_types}. "
                         f"Model {proxy_state.vllm_token_counter.model_name if proxy_state.vllm_token_counter else 'unknown'} "
                         f"is not a multimodal model.")
            return Response(
                content=json.dumps({
                    "error": {
                        "message": f"{proxy_state.vllm_token_counter.model_name if proxy_state.vllm_token_counter else 'unknown'} "
                                   f"is not a multimodal model. Detected multimodal content types: "
                                   f"{sorted(mm_part_types)}. Please remove image/audio/video inputs.",
                        "type": "BadRequestError",
                        "param": None,
                        "code": 400
                    }
                }),
                status_code=400,
                media_type="application/json",
            )
        
        # NEW: Try VLLMTokenCounter first (highest priority)
        req_body = await request.body()
        analysis = None
        request_length = len(req_body)
        
        # Debug: log which path we're taking
        logger.debug(f"vllm_token_counter available: {proxy_state.vllm_token_counter is not None}")
        logger.debug(f"Has messages: {'messages' in req_data}")
        
        if proxy_state.vllm_token_counter:
            logger.debug("Using VLLMTokenCounter for token analysis")
            try:
                if "messages" in req_data:
                    messages = req_data.get("messages", [])
                    tools = req_data.get("tools")
                    user_max_tokens = req_data.get("max_completion_tokens")
                    if user_max_tokens is None:
                        user_max_tokens = req_data.get("max_tokens")
                    user_specified_max_tokens = user_max_tokens is not None
                    
                    analysis_start = time.perf_counter()
                    # Use asyncio.to_thread to avoid blocking event loop
                    token_info = await asyncio.to_thread(
                        proxy_state.vllm_token_counter.analyze_request,
                        messages=messages,
                        tools=tools,
                    )
                    prompt_tokens = token_info['prompt_tokens']
                elif "prompt" in req_data:
                    prompt = req_data.get("prompt")
                    user_max_tokens = req_data.get("max_completion_tokens")
                    if user_max_tokens is None:
                        user_max_tokens = req_data.get("max_tokens")
                    user_specified_max_tokens = user_max_tokens is not None
                    
                    analysis_start = time.perf_counter()
                    prompt_tokens = await asyncio.to_thread(
                        proxy_state.vllm_token_counter.count_prompt_tokens,
                        prompt=prompt,
                    )
                    token_info = {
                        'prompt_tokens': prompt_tokens,
                        'system_tokens': 0,
                        'tool_tokens': 0,
                        'content_tokens': prompt_tokens,
                    }
                else:
                    # Neither messages nor prompt - skip VLLMTokenCounter analysis
                    analysis_start = 0
                    prompt_tokens = 0
                    user_max_tokens = None
                    user_specified_max_tokens = False
                    token_info = None
                
                if token_info is not None:
                    # Context length validation matching vLLM's _token_len_check:
                    # When user doesn't specify max_tokens, vLLM uses max_output_tokens=0,
                    # only checking prompt ≤ max_model_len. We mirror this behavior.
                    margin = 1 + proxy_state.context_length_margin / 100.0
                    
                    if user_specified_max_tokens:
                        # User specified max_tokens: compute effective and check total
                        model_max_tokens = proxy_state.max_model_len - prompt_tokens
                        effective_max_tokens = model_max_tokens
                        if user_max_tokens < effective_max_tokens:
                            effective_max_tokens = user_max_tokens
                        if proxy_state.override_max_tokens is not None and proxy_state.override_max_tokens < effective_max_tokens:
                            effective_max_tokens = proxy_state.override_max_tokens
                        
                        total = prompt_tokens + effective_max_tokens
                        exceeds = total * margin > proxy_state.max_model_len
                        exceeded_by = int(total * margin) - proxy_state.max_model_len if exceeds else 0
                    else:
                        # No max_tokens specified: only check prompt length (matching vLLM)
                        effective_max_tokens = 0
                        total = prompt_tokens
                        exceeds = prompt_tokens * margin > proxy_state.max_model_len
                        exceeded_by = int(prompt_tokens * margin) - proxy_state.max_model_len if exceeds else 0
                    
                    analysis_end = time.perf_counter()
                    analysis_time_ms = (analysis_end - analysis_start) * 1000
                    
                    logger.debug(f"analyze_request succeeded: prompt={prompt_tokens}, sys={token_info['system_tokens']}, tools={token_info['tool_tokens']}, content={token_info['content_tokens']}, time={analysis_time_ms:.1f}ms")
                    
                    # Create RequestAnalysis with detailed breakdown
                    analysis = RequestAnalysis()
                    analysis.prompt_tokens = prompt_tokens
                    analysis.max_tokens = effective_max_tokens
                    analysis.total_tokens = total
                    analysis.exceeds_limit = exceeds
                    analysis.exceeded_by = exceeded_by
                    analysis.analysis_time_ms = analysis_time_ms
                    analysis.system_tokens = token_info['system_tokens']
                    analysis.tool_tokens = token_info['tool_tokens']
                    analysis.content_tokens = token_info['content_tokens']
                    
                    if exceeds:
                        release_request_count()
                        max_gen_display = user_max_tokens if user_specified_max_tokens else "default"
                        logger.info(f"[REJECTED] VLLMTokenCounter: prompt={prompt_tokens}, max_gen={max_gen_display}, effective={effective_max_tokens}, "
                              f"total={total}, margin={proxy_state.context_length_margin}% → {int(total * margin)} > limit={proxy_state.max_model_len}")
                        
                        inflated_total = int(total * margin)
                        if user_specified_max_tokens:
                            inflated_prompt = inflated_total - effective_max_tokens
                            err_msg = (f"This model's maximum context length is "
                                       f"{proxy_state.max_model_len} tokens. However, you requested "
                                       f"{effective_max_tokens} output tokens and your prompt "
                                       f"contains {inflated_prompt} input tokens, "
                                       f"for a total of {inflated_total} tokens. "
                                       f"Please reduce the length of the input prompt or the "
                                       f"number of requested output tokens.")
                        else:
                            err_msg = (f"This model's maximum context length is "
                                       f"{proxy_state.max_model_len} tokens. However, your prompt "
                                       f"contains {inflated_total} input tokens, "
                                       f"which exceeds the maximum context length. "
                                       f"Please reduce the length of the input prompt.")
                        
                        return Response(
                            content=json.dumps({
                                "error": {
                                    "message": err_msg,
                                    "type": "BadRequestError",
                                    "param": "input_tokens",
                                    "code": 400
                                }
                            }),
                            status_code=400,
                            media_type="application/json",
                        )
                    
            except Exception as e:
                logger.error(f"VLLMTokenCounter analyze_request failed: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                analysis = None
        
        # Fallback: Tokenizer analysis (no quick estimation - use precise tokenizer)
        if not analysis and proxy_state.tokenizer_analyzer:
            logger.debug("Falling back to TokenizerAnalyzer")
            try:
                analysis_start = time.perf_counter()
                analysis = await asyncio.wait_for(
                    proxy_state.tokenizer_analyzer.analyze_request_async(req_data),
                    timeout=5.0
                )
                analysis_end = time.perf_counter()
                analysis.analysis_time_ms = (analysis_end - analysis_start) * 1000
                
                if analysis.exceeds_limit:
                    release_request_count()
                    
                    user_max_tokens = req_data.get("max_completion_tokens")
                    if user_max_tokens is None:
                        user_max_tokens = req_data.get("max_tokens")
                    user_specified_max_tokens = user_max_tokens is not None
                    max_gen_display = user_max_tokens if user_specified_max_tokens else "default"
                    margin = 1 + proxy_state.context_length_margin / 100.0
                    
                    if analysis.system_tokens > 0 or analysis.tool_tokens > 0:
                        logger.info(f"[REJECTED] analysis={analysis.analysis_time_ms:.1f}ms, prompt={analysis.prompt_tokens}, max_gen={max_gen_display}, "
                              f"sys={analysis.system_tokens}, tools={analysis.tool_tokens}, content={analysis.content_tokens}, "
                              f"total={analysis.total_tokens} > limit={proxy_state.max_model_len} (+{analysis.exceeded_by})")
                    else:
                        logger.info(f"[REJECTED] analysis={analysis.analysis_time_ms:.1f}ms, prompt={analysis.prompt_tokens}, max_gen={max_gen_display}, "
                              f"total={analysis.total_tokens} > limit={proxy_state.max_model_len} (+{analysis.exceeded_by})")
                    
                    inflated_total = int(analysis.total_tokens * margin)
                    if user_specified_max_tokens:
                        inflated_prompt = inflated_total - analysis.max_tokens
                        err_msg = (f"This model's maximum context length is "
                                   f"{proxy_state.max_model_len} tokens. However, you requested "
                                   f"{analysis.max_tokens} output tokens and your prompt "
                                   f"contains {inflated_prompt} input tokens, "
                                   f"for a total of {inflated_total} tokens. "
                                   f"Please reduce the length of the input prompt or the "
                                   f"number of requested output tokens.")
                    else:
                        err_msg = (f"This model's maximum context length is "
                                   f"{proxy_state.max_model_len} tokens. However, your prompt "
                                   f"contains {inflated_total} input tokens, "
                                   f"which exceeds the maximum context length. "
                                   f"Please reduce the length of the input prompt.")
                    
                    return Response(
                        content=json.dumps({
                            "error": {
                                "message": err_msg,
                                "type": "BadRequestError",
                                "param": "input_tokens",
                                "code": 400
                            }
                        }),
                        status_code=400,
                        media_type="application/json"
                    )
            except asyncio.TimeoutError:
                logger.warning(f"Tokenizer analysis timeout (>5s)")
                release_request_count()
                return Response(
                    content=json.dumps({"error": "Tokenizer analysis timeout"}),
                    status_code=500,
                    media_type="application/json"
                )
            except Exception as e:
                logger.warning(f"Tokenizer analysis failed: {e}")
                release_request_count()
                return Response(
                    content=json.dumps({"error": f"Tokenizer analysis failed: {e}"}),
                    status_code=500,
                    media_type="application/json"
                )
        
        # Load balance strategy selection
        # When analysis is None (e.g. multimodal rejected, VLLMCounter failed for
        # non-messages format, or both analyzers unavailable), use a rough estimate
        # based on content character count instead of raw byte length. Byte length
        # includes JSON structure overhead and can be orders of magnitude larger
        # than token count, distorting load balance scores.
        if global_args.use_original_lb:
            request_length = len(req_body)
        else:
            if analysis:
                request_length = analysis.prompt_tokens
            else:
                # Rough estimate: character count / 4 for text content
                # This is much closer to token count than raw byte length
                content_chars = 0
                if "messages" in req_data:
                    for msg in req_data.get("messages", []):
                        c = msg.get("content", "")
                        if isinstance(c, str):
                            content_chars += len(c)
                        elif isinstance(c, list):
                            for part in c:
                                if isinstance(part, dict) and "text" in part:
                                    content_chars += len(part["text"])
                elif "prompt" in req_data:
                    prompt = req_data.get("prompt")
                    if isinstance(prompt, str):
                        content_chars = len(prompt)
                    elif isinstance(prompt, list):
                        content_chars = sum(len(p) for p in prompt if isinstance(p, str))
                request_length = max(content_chars // 4, 1)

        if proxy_state.enable_prefix_affinity_routing and not prefix_anchors:
            prefix_anchors = _automatic_prefix_affinity_anchors(
                req_body, request_length
            )
        
        instance_info = await _handle_select_instance(
            api,
            req_data,
            request_length,
            analysis,
            prefix_anchors=prefix_anchors,
        )
        released_kv = False
        
        original_request_id = instance_info.request_id
        stream_flag = bool(req_data.get("stream", False))

        if not stream_flag:
            retry_count = 0
            while True:
                api_dp_rank = (
                    instance_info.reservation.api_dp_rank
                    if instance_info.reservation
                    else None
                )
                response = await asyncio.wait_for(
                    instance_info.decoder.client.post(
                        api,
                        json=req_data,
                        headers=_decoder_headers(
                            instance_info.request_id, api_dp_rank
                        ),
                        timeout=_http_timeout(global_args.backend_request_timeout),
                    ),
                    timeout=global_args.backend_request_timeout,
                )
                response.raise_for_status()
                if not released_kv:
                    proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                    released_kv = True
                try:
                    response_json = response.json()
                except (TypeError, ValueError):
                    response_json = None
                if not isinstance(response_json, dict):
                    return Response(
                        content=json.dumps({"error": "Decoder returned malformed JSON"}),
                        status_code=502,
                        media_type="application/json",
                    )
                choices = response_json.get("choices") or []
                choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                stop_reason = choice.get("stop_reason") or choice.get("finish_reason")
                if stop_reason not in {"recomputed", "force_free_recomputed"}:
                    content_type = response.headers.get("content-type", "application/json")
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers={"content-type": content_type},
                    )
                retry_count += 1
                if retry_count > MAX_RECOMPUTE_RETRIES:
                    return Response(
                        content=json.dumps({"error": "Decoder recompute limit exceeded"}),
                        status_code=502,
                        media_type="application/json",
                    )
                _release_decoder_reservation(instance_info)
                instance_info = await _handle_select_instance(
                    api,
                    req_data,
                    request_length,
                    analysis,
                    original_request_id=original_request_id,
                    prefix_anchors=prefix_anchors,
                )
                released_kv = False
        
        async def generate_stream():
            nonlocal instance_info, released_kv
            generated_token = ""
            retry_count = 0
            retry = True
            completion_tokens = 0
            final_stop_reason = None
            has_tool_calls = False
            has_reasoning = False
            final_usage_completion_tokens = None
            response_chunks_raw = []
            empty_delta_count = 0
            attempt_event_forwarded = False
            decoder_events = None
            try:
                while retry:
                    retry = False
                    decoder_events = stream_decoder_response(
                        instance_info.decoder.client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        decoder_api_dp_rank=(
                            instance_info.reservation.api_dp_rank
                            if instance_info.reservation
                            else None
                        ),
                    )
                    async for chunk in decoder_events:
                        if not released_kv and chunk:
                            proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                            released_kv = True
                        response_chunks_raw.append(chunk)
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.warning(f"[PARSE] {original_request_id}: UnicodeDecodeError, parse skipped (chunk still forwarded): {repr(chunk[:100])}")
                            attempt_event_forwarded = True
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: ") :]
                        if chunk_str == "[DONE]":
                            attempt_event_forwarded = True
                            yield chunk
                            continue
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            logger.warning(f"[PARSE] {original_request_id}: JSONDecodeError, parse skipped (chunk still forwarded): {chunk_str[:200]}")
                            attempt_event_forwarded = True
                            yield chunk
                            continue
                        if not isinstance(chunk_json, dict):
                            logger.warning(f"[PARSE] {original_request_id}: chunk_json is not dict (type={type(chunk_json).__name__}), parse skipped (chunk still forwarded): {chunk_str[:200]}")
                            attempt_event_forwarded = True
                            yield chunk
                            continue
                        choices = chunk_json.get("choices") or []
                        if not choices or not isinstance(choices[0], dict):
                            usage = chunk_json.get("usage") or {}
                            if usage.get("completion_tokens") is not None:
                                final_usage_completion_tokens = usage.get("completion_tokens")
                            if chunk_json:
                                logger.debug(f"[PARSE] {original_request_id}: no valid choices, extracted usage: {usage}")
                            attempt_event_forwarded = True
                            yield chunk
                            continue
                        
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
                        generated_token += content

                        delta_is_empty = (
                            delta.get("role") is None
                            and delta.get("content") is None
                            and delta.get("tool_calls") is None
                            and delta.get("function_call") is None
                            and delta.get("reasoning_content") is None
                            and delta.get("reasoning") is None
                        )
                        if delta_is_empty:
                            empty_delta_count += 1
                            if empty_delta_count == 1:
                                logger.debug(
                                    f"[EMPTY_DELTA] {original_request_id}: "
                                    f"first empty delta encountered, finish_reason={choice.get('finish_reason')}, "
                                    f"stop_reason={choice.get('stop_reason')}, "
                                    f"raw_chunk_preview={chunk_str[:200]}"
                                )

                        if delta.get("tool_calls") is not None or message.get("tool_calls") is not None:
                            has_tool_calls = True
                        if delta.get("reasoning") is not None or delta.get("reasoning_content") is not None or message.get("reasoning") is not None or message.get("reasoning_content") is not None:
                            has_reasoning = True
                        
                        stop_reason = choice.get("stop_reason") or choice.get("finish_reason")
                        usage = chunk_json.get("usage") or {}
                        if usage.get("completion_tokens") is not None:
                            final_usage_completion_tokens = usage.get("completion_tokens")
                        completion_tokens += 1
                        if stop_reason and stop_reason not in ("recomputed", "force_free_recomputed"):
                            final_stop_reason = stop_reason
                        if stop_reason in ("recomputed", "force_free_recomputed"):
                            retry_count += 1
                            logger.warning(
                                f"[RECOMPUTE] {original_request_id}: "
                                f"retry_count={retry_count}/{MAX_RECOMPUTE_RETRIES}, "
                                f"generated={len(generated_token)} chars, "
                                f"D={instance_info.decoder.url}, "
                                f"kv_in_req={bool(req_data.get('kv_transfer_params'))}, "
                                f"output_tokens_so_far={final_usage_completion_tokens}"
                            )
                            if attempt_event_forwarded:
                                yield _sse_error(
                                    "decoder_recompute_error",
                                    "Decoder requested recompute after output was sent",
                                )
                                yield b"data: [DONE]\n\n"
                                return
                            if retry_count > MAX_RECOMPUTE_RETRIES:
                                logger.error(
                                    f"[RECOMPUTE] {original_request_id}: "
                                    f"max retries ({MAX_RECOMPUTE_RETRIES}) exceeded, giving up"
                                )
                                yield _sse_error(
                                    "decoder_recompute_error",
                                    "Decoder recompute limit exceeded",
                                )
                                yield b"data: [DONE]\n\n"
                                return
                            retry = True
                            
                            # Release old P/D resources before recompute replaces instance_info.
                            if not released_kv:
                                proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                                released_kv = True
                            _release_decoder_reservation(instance_info)
                            
                            generated_token = ""
                            completion_tokens = 0
                            final_stop_reason = None
                            has_tool_calls = False
                            has_reasoning = False
                            final_usage_completion_tokens = None
                            response_chunks_raw = []
                            empty_delta_count = 0
                            attempt_event_forwarded = False
                            instance_info = await _handle_select_instance(
                                api,
                                req_data,
                                request_length,
                                analysis,
                                original_request_id=original_request_id,
                                prefix_anchors=prefix_anchors,
                            )
                            released_kv = False
                            logger.info(
                                f"[RECOMPUTE] {original_request_id}: "
                                f"new_req_id={instance_info.request_id}"
                            )
                            break
                        attempt_event_forwarded = True
                        yield chunk
                    await decoder_events.aclose()
                    decoder_events = None
            except Exception as e:
                logger.error(
                    f"Error during streaming from decoder {instance_info.decoder.url}: {str(e)} "
                    f"the aborted request {instance_info.request_id} will be routing to the target "
                    "prefiller when new request is ready to dispatch to it"
                )
                proxy_state.abort_prefiller_request(instance_info.prefiller_idx, instance_info.request_id)
                if not released_kv:
                    proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                    released_kv = True
                if isinstance(e, (asyncio.TimeoutError, httpx.TimeoutException)):
                    error_code = "decoder_timeout"
                    error_message = "Decoder stream timed out"
                elif isinstance(e, _IncompleteDecoderStreamError):
                    error_code = "decoder_stream_incomplete"
                    error_message = "Decoder stream ended before completion"
                else:
                    error_code = "decoder_backend_error"
                    error_message = "Decoder stream interrupted"
                yield _sse_error(error_code, error_message)
                yield b"data: [DONE]\n\n"
            finally:
                try:
                    if decoder_events is not None:
                        await decoder_events.aclose()
                finally:
                    cleanup_current_request()
                if empty_delta_count > 0:
                    logger.debug(
                        f"[EMPTY_DELTA_SUMMARY] {original_request_id}: "
                        f"empty_delta_count={empty_delta_count}, "
                        f"generated_token_len={len(generated_token)}, "
                        f"output_tokens={final_usage_completion_tokens}, "
                        f"stop_reason={final_stop_reason}, "
                        f"has_tool_calls={has_tool_calls}, "
                        f"has_reasoning={has_reasoning}"
                    )
                is_empty_response = (
                    not generated_token
                    and not has_tool_calls
                    and not has_reasoning
                    and (final_stop_reason is not None or retry_count > MAX_RECOMPUTE_RETRIES)
                )
                if is_empty_response:
                    logger.warning(
                        f"[EMPTY_RESPONSE] {instance_info.request_id}: "
                        f"stop_reason={final_stop_reason}, "
                        f"output_tokens={final_usage_completion_tokens}, "
                        f"has_tool_calls={has_tool_calls}, "
                        f"has_reasoning={has_reasoning}, "
                        f"P={instance_info.prefiller.url}, D={instance_info.decoder.url}, "
                        f"kv_in_req={bool(req_data.get('kv_transfer_params'))}, "
                        f"retry_count={retry_count}"
                    )
                    try:
                        save_dir = os.path.dirname(os.path.abspath(__file__))
                        save_path = os.path.join(save_dir, f"{instance_info.request_id}.jsonl")
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "type": "meta",
                                "request_id": instance_info.request_id,
                                "stop_reason": final_stop_reason,
                                "output_tokens": final_usage_completion_tokens,
                                "has_tool_calls": has_tool_calls,
                                "generated_token": generated_token,
                                "completion_tokens": completion_tokens,
                                "retry_count": retry_count,
                                "kv_in_req": bool(req_data.get("kv_transfer_params")),
                                "prefiller": str(instance_info.prefiller.url),
                                "decoder": str(instance_info.decoder.url),
                            }, ensure_ascii=False) + "\n")
                            f.write(json.dumps({
                                "type": "request",
                                "body": req_data,
                            }, ensure_ascii=False) + "\n")
                            for c in response_chunks_raw:
                                f.write(json.dumps({
                                    "type": "chunk",
                                    "raw": c.decode("utf-8", errors="replace"),
                                }, ensure_ascii=False) + "\n")
                        logger.info(f"[EMPTY_RESPONSE] Full response data saved to {save_path}")
                    except Exception as save_err:
                        logger.error(f"[EMPTY_RESPONSE] Failed to save response data: {save_err}")

                is_empty_delta_with_output = (
                    empty_delta_count > 1
                    and (final_usage_completion_tokens is not None and final_usage_completion_tokens > 0)
                    and not generated_token
                    and not has_tool_calls
                    and not has_reasoning
                )
                if is_empty_delta_with_output:
                    logger.debug(
                        f"[EMPTY_DELTA_RESPONSE] {original_request_id}: "
                        f"empty_delta_count={empty_delta_count}, "
                        f"output_tokens={final_usage_completion_tokens}, "
                        f"generated_token_len={len(generated_token)}, "
                        f"stop_reason={final_stop_reason}, "
                        f"has_tool_calls={has_tool_calls}, "
                        f"has_reasoning={has_reasoning}, "
                        f"P={instance_info.prefiller.url}, D={instance_info.decoder.url}, "
                        f"kv_in_req={bool(req_data.get('kv_transfer_params'))}"
                    )
                    try:
                        save_dir = os.path.dirname(os.path.abspath(__file__))
                        save_path = os.path.join(save_dir, f"{original_request_id}_empty_delta.jsonl")
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "type": "meta",
                                "request_id": original_request_id,
                                "original_request_id": original_request_id,
                                "proxy_request_id": instance_info.request_id,
                                "empty_delta_count": empty_delta_count,
                                "output_tokens": final_usage_completion_tokens,
                                "generated_token_len": len(generated_token),
                                "stop_reason": final_stop_reason,
                                "has_tool_calls": has_tool_calls,
                                "has_reasoning": has_reasoning,
                                "retry_count": retry_count,
                                "kv_in_req": bool(req_data.get("kv_transfer_params")),
                                "prefiller": str(instance_info.prefiller.url),
                                "decoder": str(instance_info.decoder.url),
                            }, ensure_ascii=False) + "\n")
                            f.write(json.dumps({
                                "type": "request",
                                "body": req_data,
                            }, ensure_ascii=False) + "\n")
                            for c in response_chunks_raw:
                                f.write(json.dumps({
                                    "type": "chunk",
                                    "raw": c.decode("utf-8", errors="replace"),
                                }, ensure_ascii=False) + "\n")
                        logger.info(f"[EMPTY_DELTA_RESPONSE] Full response data saved to {save_path}")
                    except Exception as save_err:
                        logger.error(f"[EMPTY_DELTA_RESPONSE] Failed to save response data: {save_err}")

                is_reasoning_only_response = (
                    has_reasoning
                    and not generated_token
                    and not has_tool_calls
                    and final_stop_reason is not None
                )
                if is_reasoning_only_response:
                    logger.warning(
                        f"[REASONING_ONLY_RESPONSE] {original_request_id}: "
                        f"stop_reason={final_stop_reason}, "
                        f"output_tokens={final_usage_completion_tokens}, "
                        f"has_tool_calls={has_tool_calls}, "
                        f"has_reasoning={has_reasoning}, "
                        f"empty_delta_count={empty_delta_count}, "
                        f"P={instance_info.prefiller.url}, D={instance_info.decoder.url}, "
                        f"kv_in_req={bool(req_data.get('kv_transfer_params'))}, "
                        f"retry_count={retry_count}"
                    )
                    try:
                        save_dir = os.path.dirname(os.path.abspath(__file__))
                        save_path = os.path.join(save_dir, f"{original_request_id}_reasoning_only.jsonl")
                        with open(save_path, "w", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "type": "meta",
                                "request_id": original_request_id,
                                "original_request_id": original_request_id,
                                "proxy_request_id": instance_info.request_id,
                                "stop_reason": final_stop_reason,
                                "output_tokens": final_usage_completion_tokens,
                                "has_tool_calls": has_tool_calls,
                                "has_reasoning": has_reasoning,
                                "empty_delta_count": empty_delta_count,
                                "generated_token_len": len(generated_token),
                                "retry_count": retry_count,
                                "kv_in_req": bool(req_data.get("kv_transfer_params")),
                                "prefiller": str(instance_info.prefiller.url),
                                "decoder": str(instance_info.decoder.url),
                            }, ensure_ascii=False) + "\n")
                            f.write(json.dumps({
                                "type": "request",
                                "body": req_data,
                            }, ensure_ascii=False) + "\n")
                            for c in response_chunks_raw:
                                f.write(json.dumps({
                                    "type": "chunk",
                                    "raw": c.decode("utf-8", errors="replace"),
                                }, ensure_ascii=False) + "\n")
                        logger.info(f"[REASONING_ONLY_RESPONSE] Full response data saved to {save_path}")
                    except Exception as save_err:
                        logger.error(f"[REASONING_ONLY_RESPONSE] Failed to save response data: {save_err}")
        
        # Determine the correct media type based on stream flag
        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        response = _CleanupStreamingResponse(
            generate_stream(),
            media_type=media_type,
            cleanup=cleanup_current_request,
        )
        response_owns_cleanup = True
        return response
    
    except asyncio.TimeoutError:
        logger.error("Backend model request timed out")
        return Response(
            content=json.dumps({"error": "Backend model request timed out"}),
            status_code=504,
            media_type="application/json",
        )

    except httpx.HTTPStatusError as e:
        # Backend returned error status code - propagate to client
        logger.error(f"Backend returned error status {e.response.status_code}: {str(e)}")
        
        # Try to get error body from backend
        try:
            error_body = e.response.text
        except Exception:
            error_body = json.dumps({"error": f"Backend error: {e.response.status_code}"})
        
        return Response(
            content=error_body,
            status_code=e.response.status_code,
            media_type="application/json"
        )
    
    except httpx.RequestError as e:
        # Network error - backend unavailable
        logger.error(f"Backend unavailable: {str(e)}")
        
        return Response(
            content=json.dumps({"error": "Backend service unavailable"}),
            status_code=503,
            media_type="application/json"
        )
    
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        logger.info(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        logger.error(str(e))
        logger.info("".join(traceback.format_exception(*exc_info)))
        
        return Response(
            content=json.dumps({"error": f"Internal proxy error: {str(e)}"}),
            status_code=500,
            media_type="application/json"
        )
    finally:
        if not response_owns_cleanup:
            cleanup_current_request()


async def _handle_adjust_instances(adjust_mode: str, request: Request):
    try:
        req_data = await request.json()
        instance_type = req_data.get("type", "")
        instances = req_data.get("instances", [])
        if isinstance(instances, str):
            instances = [instances]
        if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
            }
        if proxy_state.enable_remote_lmcache_store:
            return {
                "error": "Dynamic topology is disabled in placement-aware mode; restart the paired proxy/P/D topology."
            }
        instances = trans_instances(instances)
        all_msg = f"{adjust_mode} {instance_type} instances: {[str(server) for server in instances]}."
        
        if adjust_mode == "add":
            added_nodes, waiting_nodes = await proxy_state.add_instances(instance_type, instances)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(instances)
            else:
                need_waiting = proxy_state.remove_decoders(instances)
            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."
        
        # Enhanced: update metrics aggregator reference
        if proxy_state.metrics_aggregator:
            proxy_state.metrics_aggregator.prefillers = proxy_state.prefillers
            proxy_state.metrics_aggregator.decoders = proxy_state.decoders
        
        return {
            "message": all_msg,
            "current_prefill_instances": [str(prefiller) for prefiller in proxy_state.prefillers],
            "current_decode_instances": [str(decoder) for decoder in proxy_state.decoders],
        }
    except Exception as e:
        logger.error(f"Failed to {adjust_mode} instances: {e}")
        raise e


def trans_instances(instances: list[str]) -> list[ServerState]:
    server_list = []
    for instance in instances:
        h, p = instance.split(":")
        server_list.append(ServerState(h, int(p)))
    return server_list


# ============================================================================
# FastAPI App Setup
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument(
        "--enable-remote-lmcache-store",
        action="store_true",
        help="Enable decoder RemoteFill discovery and direct remote LMCache storage",
    )
    parser.add_argument(
        "--enable-prefix-affinity-routing",
        action="store_true",
        help=(
            "Route bounded X-LMCache-Prefix-Affinity token_end=opaque_key hints "
            "to their learned P/D placement"
        ),
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries")
    parser.add_argument("--retry-delay", type=float, default=0.001, help="Base delay for exponential backoff")
    parser.add_argument(
        "--backend-request-timeout",
        type=float,
        default=_BACKEND_REQUEST_TIMEOUT_SECONDS,
        help="Total prefiller and non-stream decoder timeout in seconds",
    )
    parser.add_argument(
        "--decoder-read-timeout",
        type=float,
        default=_DECODER_READ_TIMEOUT_SECONDS,
        help="Decoder streaming read-idle timeout in seconds",
    )
    parser.add_argument("--max-waiting-retries", type=int, default=3, help="Maximum retries for waiting nodes")
    parser.add_argument("--waiting-retry-interval", type=float, default=10, help="Check interval for waiting nodes")
    # Enhanced arguments
    parser.add_argument("--model-name", type=str, default=None, help="Model name for tokenizer and metrics")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Maximum model context length")
    parser.add_argument("--chat-template", type=str, default=None, 
                        help="Path to custom chat template file for tokenizer (e.g., for Code Agent scenarios)")
    parser.add_argument("--disable-tokenizer-analysis", action="store_true",
                        help="Disable exact tokenization and use approximate load accounting")
    parser.add_argument("--disable-metrics", action="store_true", help="Disable metrics polling")
    parser.add_argument("--disable-metrics-polling", action="store_true", 
                        help="Disable background metrics polling only (API endpoints still available)")
    parser.add_argument("--metrics-poll-interval", type=float, default=5.0, help="Metrics polling interval")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for tokenizer")
    parser.add_argument("--use-original-lb", action="store_true", 
                        help="Use original load balance logic (byte-based) instead of enhanced token-based")
    parser.add_argument("--default-max-tokens", type=int, default=None,
                        help="Default max_tokens when user doesn't specify (fallback if VLLMTokenCounter unavailable)")
    parser.add_argument("--override-max-tokens", type=int, default=None,
                        help="Override max_tokens cap from generation_config (fallback if VLLMTokenCounter unavailable)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (DEBUG for verbose, INFO for normal)")
    parser.add_argument("--context-length-margin", type=int, default=5,
                        help="Safety margin percentage for context length check: "
                             "reject if computed_length * (1 + margin%%) > max_model_len. "
                             "Default 5%% to account for chat template overhead estimation error.")
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    if min(
        args.backend_request_timeout,
        args.decoder_read_timeout,
    ) <= 0:
        raise ValueError("Backend timeout values must be positive")
    if args.enable_prefix_affinity_routing and not args.enable_remote_lmcache_store:
        raise ValueError(
            "--enable-prefix-affinity-routing requires "
            "--enable-remote-lmcache-store"
        )
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


proxy_state = None
global_args = None


async def listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        try:
            done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            handler_task.cancel()
            cancellation_task.cancel()
            await asyncio.gather(
                handler_task, cancellation_task, return_exceptions=True
            )
            raise
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        with suppress(asyncio.CancelledError):
            await handler_task
        return None
    return wrapper


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state, global_args
    
    tokenizer_analyzer = None
    metrics_aggregator = None
    vllm_token_counter = None
    default_max_tokens = None
    override_max_tokens = None
    
    # NEW: Initialize VLLMTokenCounter (highest priority)
    if (
        global_args.model_name
        and VLLM_TOKEN_COUNTER_AVAILABLE
        and not global_args.disable_tokenizer_analysis
    ):
        try:
            vllm_token_counter = VLLMTokenCounter(
                model_name=global_args.model_name,
                max_model_len=global_args.max_model_len,
                chat_template=global_args.chat_template,
                trust_remote_code=global_args.trust_remote_code,
            )
            logger.info(f"VLLMTokenCounter initialized: model={global_args.model_name}")
        except Exception as e:
            logger.info(f"VLLMTokenCounter initialization failed: {e}")
            vllm_token_counter = None
    
    # Fallback: Initialize TokenizerAnalyzer
    if not vllm_token_counter and global_args.model_name and TOKENIZER_ENABLED and not global_args.disable_tokenizer_analysis:
        try:
            tokenizer_analyzer = TokenizerAnalyzer(
                model_name=global_args.model_name,
                max_model_len=global_args.max_model_len,
                trust_remote_code=global_args.trust_remote_code,
                chat_template=global_args.chat_template,
                default_max_tokens=global_args.default_max_tokens if hasattr(global_args, 'default_max_tokens') and global_args.default_max_tokens else global_args.max_model_len,
                override_max_tokens=global_args.override_max_tokens if hasattr(global_args, 'override_max_tokens') else None,
                context_length_margin=global_args.context_length_margin,
            )
            logger.info(f"TokenizerAnalyzer initialized: model={global_args.model_name}, max_len={global_args.max_model_len}")
        except Exception as e:
            logger.info(f"Tokenizer initialization failed: {e}")
            tokenizer_analyzer = None
    else:
        if global_args.disable_tokenizer_analysis:
            logger.info("Tokenizer analysis disabled by --disable-tokenizer-analysis")
        elif not global_args.model_name:
            logger.info("Tokenizer not enabled: --model-name not provided")
        elif not TOKENIZER_ENABLED:
            logger.info("Tokenizer not available: transformers not installed")
    
    # NEW: Get default_max_tokens and override_max_tokens from P node config
    if vllm_token_counter:
        default_max_tokens = vllm_token_counter.get_default_max_tokens()
        override_max_tokens = vllm_token_counter.get_override_max_tokens()
        logger.info(f"P node max_tokens config: default={default_max_tokens}, override={override_max_tokens}")
    elif tokenizer_analyzer:
        default_max_tokens = global_args.default_max_tokens if hasattr(global_args, 'default_max_tokens') and global_args.default_max_tokens else global_args.max_model_len
        override_max_tokens = global_args.override_max_tokens if hasattr(global_args, 'override_max_tokens') else None
        logger.info(f"P node max_tokens config (fallback): default={default_max_tokens}, override={override_max_tokens}")
    
    if global_args.model_name and not global_args.disable_metrics:
        metrics_aggregator = MetricsAggregator(
            prefillers=[],
            decoders=[],
            model_name=global_args.model_name,
            poll_interval=global_args.metrics_poll_interval,
            enable_polling=not global_args.disable_metrics_polling,
        )
    
    proxy_state = ProxyState(
        global_args.prefiller_instances,
        global_args.decoder_instances,
        tokenizer_analyzer=tokenizer_analyzer,
        metrics_aggregator=metrics_aggregator,
        max_model_len=global_args.max_model_len,
        vllm_token_counter=vllm_token_counter,
        default_max_tokens=default_max_tokens,
        override_max_tokens=override_max_tokens,
        context_length_margin=global_args.context_length_margin,
        enable_remote_lmcache_store=global_args.enable_remote_lmcache_store,
        enable_prefix_affinity_routing=getattr(
            global_args, "enable_prefix_affinity_routing", False
        ),
    )
    
    # Enhanced: set metrics aggregator references
    if metrics_aggregator:
        metrics_aggregator.prefillers = proxy_state.prefillers
        metrics_aggregator.decoders = proxy_state.decoders
        metrics_aggregator.start()
    
    logger.info(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    if proxy_state.enable_prefix_affinity_routing:
        logger.info(
            "Prefix-affinity routing enabled via %s or automatic request fingerprint",
            _PREFIX_AFFINITY_HEADER,
        )
    
    # Print load balance mode
    if global_args.use_original_lb:
        logger.info("[INFO] Using ORIGINAL load balance logic (byte-based)")
    else:
        logger.info("[INFO] Using ENHANCED load balance logic (token-based)")
        if proxy_state.vllm_token_counter or proxy_state.tokenizer_analyzer:
            logger.info(f"[INFO] Exact tokenizer analysis enabled: model={global_args.model_name}")
        else:
            logger.info("[INFO] Exact tokenizer analysis disabled; using character estimate")
    
    yield
    
    # Cleanup
    if metrics_aggregator:
        metrics_aggregator.stop()
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


app = FastAPI(lifespan=lifespan)


# ============================================================================
# Endpoints (from original + enhanced)
# ============================================================================

@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
    }


@app.get("/stats")
async def stats():
    """Enhanced: Detailed stats"""
    return proxy_state.get_stats()


@app.get("/backend_metrics")
async def backend_metrics():
    """Enhanced: Backend metrics aggregation"""
    if proxy_state.metrics_aggregator:
        return await proxy_state.metrics_aggregator.fetch_all_metrics()
    return {"error": "Metrics aggregator not enabled"}


@app.get("/metrics")
async def metrics_endpoint():
    """Enhanced: Prometheus metrics - vLLM native format"""
    if not proxy_state.metrics_aggregator:
        return Response(content="# Metrics aggregator not enabled\n", media_type="text/plain")
    await proxy_state.metrics_aggregator.fetch_all_metrics()
    output = proxy_state.metrics_aggregator.generate_vllm_metrics_output()
    return Response(content=output, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


if __name__ == "__main__":
    global_args = parse_args()
    logging.getLogger().setLevel(getattr(logging, global_args.log_level, logging.INFO))
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
