# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _load_helpers():
    """Execute the path helpers without running setup() at module scope."""
    tree = ast.parse((ROOT / "setup.py").read_text())
    names = {"is_editable_build", "get_cmake_install_paths"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    module = ast.parse("")
    module.body = nodes
    namespace = {"os": os}
    exec(compile(module, str(ROOT / "setup.py"), "exec"), namespace)
    return namespace


def _build(build_lib="build/lib.test", *, inplace=False, editable_mode=False, commands=None):
    distribution = SimpleNamespace()
    if commands is not None:
        distribution.commands = commands
    return SimpleNamespace(build_lib=build_lib, inplace=inplace, editable_mode=editable_mode, distribution=distribution)


def test_non_editable_build_uses_build_lib():
    helpers = _load_helpers()
    for commands in (None, (), ("build_ext",)):
        build_ext_cmd = _build(commands=commands)
        install_path, package_path = helpers["get_cmake_install_paths"](build_ext_cmd, str(ROOT))

        assert install_path == os.path.join(str(ROOT), "build/lib.test")
        assert package_path == os.path.join(install_path, "vllm_ascend")


def test_editable_build_variants_install_into_source_tree():
    helpers = _load_helpers()
    source_package = os.path.join(str(ROOT), "vllm_ascend")
    builds = (
        _build(inplace=True),
        _build(editable_mode=True),
        _build(commands=("develop",)),
    )
    for build_ext_cmd in builds:
        install_path, package_path = helpers["get_cmake_install_paths"](build_ext_cmd, str(ROOT))
        assert install_path == source_package
        assert package_path == source_package
