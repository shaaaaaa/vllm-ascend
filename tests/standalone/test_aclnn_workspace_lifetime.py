# SPDX-License-Identifier: Apache-2.0
"""Compile the real ACLNN macro against a CPU stream/allocator lifetime probe.

This checks the C++ owner's lifetime and FIFO reuse, not CANN/SFA correctness.
No torch, NPU runtime, model, or server is needed.
"""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HEADER = Path(__file__).resolve().parents[2] / "csrc/aclnn_torch_adapter/op_api_common.h"

PREAMBLE = r"""
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#define TORCH_CHECK(ok, ...) \
  do { if (!(ok)) throw std::runtime_error("macro check failed"); } while (false)

int current_stream = 0;
bool queued = true;
int compared = 0;
int released = 0;
struct Block {
  int stream;
  bool owned = false;
  std::vector<unsigned char> data;
};
std::vector<std::unique_ptr<Block>> pool;
Block *prepared_block = nullptr;
std::vector<std::function<int()>> host_queue;
std::vector<std::function<void()>> device_queues[2];

// Like a caching allocator, releasing the Tensor owner makes a block available
// on its original stream; the allocation itself remains resident in the pool.
struct Lease {
  Block *block;
  ~Lease() { block->owned = false; ++released; }
};
namespace at {
struct TensorOptions {
  explicit TensorOptions(int) {}
  TensorOptions dtype(int) const { return *this; }
};
struct Storage {
  Block *block;
  const void *data() const { return block->data.data(); }
};
struct Tensor {
  std::shared_ptr<Lease> owner;
  Storage storage() const { return {owner->block}; }
};
Tensor empty(std::initializer_list<uint64_t> size, TensorOptions) {
  const auto bytes = *size.begin();
  Block *block = nullptr;
  for (auto &candidate : pool) {
    if (!candidate->owned && candidate->stream == current_stream &&
        candidate->data.size() >= bytes) {
      block = candidate.get();
      break;
    }
  }
  if (!block) {
    pool.emplace_back(new Block{current_stream, false,
                               std::vector<unsigned char>(bytes)});
    block = pool.back().get();
  }
  block->owned = true;
  prepared_block = block;
  return {std::shared_ptr<Lease>(new Lease{block})};
}
}
constexpr int kByte = 0;
namespace torch_npu { namespace utils {
int get_npu_device_type() { return 0; }
}}
using aclrtStream = int;
namespace c10_npu {
struct Stream { int stream(bool) const { return current_stream; } };
Stream getCurrentNPUStream() { return {}; }
}
struct Input { uint64_t bytes; unsigned char value; };
struct aclOpExecutor { Input input; };
int prepare(Input input, uint64_t *bytes, aclOpExecutor **executor) {
  *bytes = input.bytes;
  *executor = new aclOpExecutor{input};
  prepared_block = nullptr;
  return 0;
}
int execute(void *address, uint64_t bytes, aclOpExecutor *executor,
            const aclrtStream stream) {
  const auto value = executor->input.value;
  if (!bytes) {
    if (address) throw std::runtime_error("zero workspace must be null");
    ++compared;
  } else {
    auto *data = static_cast<unsigned char *>(address);
    device_queues[stream].push_back([=] { std::fill_n(data, bytes, value); });
    device_queues[stream].push_back([=] {
      if (!std::all_of(data, data + bytes,
                      [=](unsigned char byte) { return byte == value; }))
        throw std::runtime_error("workspace overwritten during device use");
      ++compared;
    });
  }
  delete executor;
  return 0;
}
void *GetOpApiFuncAddr(const char *name) {
  if (std::string(name) == "aclnnProbeGetWorkspaceSize")
    return reinterpret_cast<void *>(&prepare);
  if (std::string(name) == "aclnnProbe")
    return reinterpret_cast<void *>(&execute);
  return nullptr;
}
const char *GetOpApiLibName() { return "probe"; }
const char *aclGetRecentErrMsg() { return "probe"; }
template<class... Args> auto ConvertTypes(Args... args) {
  return std::make_tuple(args...);
}
template<class T> auto ConvertToOpApiFunc(const T &, void *address) {
  return reinterpret_cast<int (*)(Input, uint64_t *, aclOpExecutor **)>(address);
}
template<class F, class T> auto call(F function, T &arguments) {
  return std::apply(function, arguments);
}
template<class T> void ReleaseConvertTypes(T &) {}
using InitHugeMemThreadLocal = int (*)(void *, bool);
using UnInitHugeMemThreadLocal = int (*)(void *, bool);
using ReleaseHugeMem = int (*)(void *, bool);
namespace at_npu { namespace native {
struct OpCommand {
  std::function<int()> handler;
  void Name(const char *) {}
  void SetCustomHandler(std::function<int()> function) { handler = function; }
  void Run() {
    if (prepared_block && !prepared_block->owned)
      throw std::runtime_error("workspace released before command enqueue");
    if (queued) host_queue.push_back(handler);
    else handler();
  }
};
}}
"""

MAIN = r"""
int main(int argc, char **argv) {
  try {
    queued = argc < 2 || std::string(argv[1]) == "queued";
    const uint64_t bytes = argc < 3 ? 64 : std::stoul(argv[2]);
    constexpr int commands = 32;
    for (int i = 0; i < commands; ++i) {
      current_stream = i % 2;
      Input input{bytes, static_cast<unsigned char>(i + 1)};
      EXEC_NPU_CMD(aclnnProbe, input);
    }
    // The host has returned from every operator, but no simulated NPU work
    // has completed yet. Queued handlers still refer to resident pool blocks.
    if (bytes && released != commands)
      throw std::runtime_error("workspace owner leaked after enqueue");
    for (auto &operation : host_queue) operation();
    for (auto &stream : device_queues)
      for (auto &operation : stream) operation();
    if (compared != commands || (bytes && pool.size() != 2))
      throw std::runtime_error("FIFO reuse or completion count failed");
    std::cout << "owners and FIFO reuse OK\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


def _compiler():
    if os.environ.get("CXX"):
        return shlex.split(os.environ["CXX"])
    for name in ("c++", "g++", "clang++"):
        if path := shutil.which(name):
            return [path]
    if importlib.util.find_spec("ziglang") is not None:
        return [sys.executable, "-m", "ziglang", "c++"]
    pytest.skip("C++ compiler required (CXX, c++, or python package ziglang)")


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    root = tmp_path_factory.mktemp("aclnn-workspace")
    header = HEADER.read_text(encoding="utf-8")
    macro = header[header.index("#define EXEC_NPU_CMD(") :].split("\n#endif", 1)[0]
    # Negative control: the original allocation block releases its owner before
    # OpCommand can enqueue the consumer. Compile both versions of the real macro.
    old = macro.replace("    at::Tensor workspace_tensor;", "    /* no outer owner */", 1)
    old = old.replace("      workspace_tensor =", "      auto workspace_tensor =", 1)
    binaries = []
    for name, source in (("fixed", macro), ("before_fix", old)):
        cpp = root / f"{name}.cpp"
        cpp.write_text(PREAMBLE + "\n" + source + "\n" + MAIN, encoding="utf-8")
        binary = root / (name + (".exe" if os.name == "nt" else ""))
        result = subprocess.run(
            [*_compiler(), "-std=c++17", str(cpp), "-o", str(binary)],
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        binaries.append(binary)
    return binaries


def _run(binary, mode, workspace_bytes):
    try:
        return subprocess.run(
            [str(binary), mode, str(workspace_bytes)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as error:
        if getattr(error, "winerror", None) == 4551:
            pytest.skip("Windows application control blocks the compiled probe (WinError 4551)")
        raise


@pytest.mark.parametrize("mode", ["queued", "immediate"])
@pytest.mark.parametrize("workspace_bytes", [0, 64, 16384])
def test_workspace_survives_enqueue_and_fifo_reuse(probes, mode, workspace_bytes):
    result = _run(probes[0], mode, workspace_bytes)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["queued", "immediate"])
def test_probe_rejects_original_premature_release(probes, mode):
    result = _run(probes[1], mode, 64)
    assert result.returncode == 1
    assert "workspace released before command enqueue" in result.stderr
