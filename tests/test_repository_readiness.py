import hashlib
from pathlib import Path
import re
import subprocess
import tomllib

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COLLABORATION_TEMPLATES = (
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/pull_request_template.md",
)
APPROVED_SECURITY_EMAIL = "opensource@limxdynamics.com"
EXTERNAL_EXAMPLE_SOURCES = (
    (
        "examples/aloha_real/README.md",
        "https://github.com/Physical-Intelligence/aloha.git",
        "d1dc83afd89ded4379851257fe5d85632d31d5ec",
        "third_party/aloha",
    ),
    (
        "examples/libero/README.md",
        "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
        "third_party/libero",
    ),
)
APPROVED_LICENSE_HASHES = {
    "LICENSES/ACT-MIT.txt": "4666c312da313e6c46929f6695d06cf98a2e7359b9c7dcbb0ea232d01b32cd42",
    "LICENSES/msgpack-numpy-BSD-3-Clause.txt": "509aa7af19ae2b5b1681b22b969627ee9e1fd15ff7ea5bf9f6e301fc44d5a6c2",
    "LICENSES/Kinetix-MIT.txt": "2314c19f6a70f084aaecded7a427b688e6e93b9016f6837f2a9a91a9e17ddf3f",
    "LICENSES/DROID-MIT.txt": "46c07e0a692891260f8fcc89a8904d4d688c5d625e891d8957597bc1915d95e6",
    "LICENSES/robosuite-MIT-and-Apache-2.0.txt": "177978cbece0a4c454c2aaec5b3f145b39270814874c43109da9e829c39d9cba",
}
OPENPI_WORKING_BASELINE = "e01d2290dfef823304b9a59a94b29e5945e38b2d"


def _read_repository_file(relative_path: str) -> str:
    path = REPOSITORY_ROOT / relative_path
    assert path.is_file(), f"missing repository readiness file: {relative_path}"
    content = path.read_text(encoding="utf-8")
    assert content.strip(), f"repository readiness file is empty: {relative_path}"
    return content


def _assert_terms(content: str, terms: tuple[str, ...]) -> None:
    normalized = " ".join(content.lower().split())
    for term in terms:
        assert term.lower() in normalized


@pytest.mark.parametrize(
    "relative_path",
    [
        "CHANGELOG.md",
        *COLLABORATION_TEMPLATES,
        ".github/workflows/ci.yml",
    ],
)
def test_repository_readiness_files_exist_and_are_nonempty(relative_path: str):
    _read_repository_file(relative_path)


def test_changelog_has_an_unreleased_section_without_placeholders():
    content = _read_repository_file("CHANGELOG.md")

    assert "# Changelog" in content
    assert "## Unreleased" in content
    assert "TODO" not in content
    assert "TBD" not in content
    _assert_terms(content, ("security reporting", "network deployment"))


def test_default_pytest_collection_includes_repository_readiness_tests():
    configuration = tomllib.loads(_read_repository_file("pyproject.toml"))
    testpaths = configuration["tool"]["pytest"]["ini_options"]["testpaths"]

    assert "tests" in testpaths


@pytest.mark.parametrize("relative_path", COLLABORATION_TEMPLATES)
def test_collaboration_templates_cover_public_contribution_boundaries(relative_path: str):
    content = _read_repository_file(relative_path).lower()

    for sensitive_term in ("credential", "customer data", "private log", "model weight"):
        assert sensitive_term in content
    assert "real robot" in content or "real-robot" in content
    assert "safety" in content
    assert "third-party" in content
    assert "license" in content


def test_ci_is_explicitly_scoped_to_python_cpu_software_checks():
    content = _read_repository_file(".github/workflows/ci.yml")
    normalized = content.lower()

    assert "ubuntu-latest" in content
    assert 'python-version: "3.11"' in content
    assert "software" in normalized
    assert "cpu" in normalized
    assert "no gpu" in normalized
    assert "no robot hardware" in normalized

    assert "python -m pip install pytest" in content
    assert "python -m pip install -e packages/openpi-client" in content
    assert "python -m pytest -q tests/test_repository_readiness.py" in content
    assert "packages/openpi-client/src/openpi_client/image_tools_test.py" in content
    assert "packages/openpi-client/src/openpi_client/msgpack_numpy_test.py" in content
    assert "python -m compileall -q ." in content


def test_security_policy_uses_the_approved_private_reporting_channel():
    content = _read_repository_file("SECURITY.md")

    assert APPROVED_SECURITY_EMAIL in content
    _assert_terms(
        content,
        (
            "private",
            "confidential",
            "public issue",
            "real credentials",
            "customer data",
            "field data",
            "private logs",
            "model weights",
        ),
    )


@pytest.mark.parametrize(
    "relative_path",
    ["CONTRIBUTING.md", ".github/ISSUE_TEMPLATE/bug_report.yml"],
)
def test_public_issue_surfaces_route_suspected_vulnerabilities_privately(relative_path: str):
    content = _read_repository_file(relative_path)

    assert APPROVED_SECURITY_EMAIL in content
    _assert_terms(content, ("suspected vulnerability", "public issue", "privately"))


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.md",
        "docs/remote_inference.md",
        "configs/deploy/tron2_deploy.server.example.yaml",
        "configs/deploy/tron2_deploy.client.example.yaml",
    ],
)
def test_english_runtime_docs_define_the_network_deployment_boundary(relative_path: str):
    content = _read_repository_file(relative_path)

    _assert_terms(
        content,
        (
            "controlled robot lan",
            "internet",
            "untrusted",
            "shared network",
            "authentication",
            "tls",
            "cross-site",
            "cloud",
            "security review",
        ),
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "README_CN.md",
        "configs/deploy/tron2_deploy.server.example_CN.yaml",
        "configs/deploy/tron2_deploy.client.example_CN.yaml",
    ],
)
def test_chinese_runtime_docs_define_the_network_deployment_boundary(relative_path: str):
    content = _read_repository_file(relative_path)

    _assert_terms(
        content,
        (
            "受控机器人局域网",
            "互联网",
            "不受信任",
            "共享网络",
            "鉴权",
            "tls",
            "跨站点",
            "云端",
            "安全评审",
        ),
    )


@pytest.mark.parametrize(
    ("relative_path", "terms"),
    [
        ("README.md", ("source disclosure", "functional safety", "real-robot certification")),
        ("README_CN.md", ("源码公开", "功能安全", "真机认证")),
        (
            "configs/deploy/tron2_deploy.server.example.yaml",
            ("source disclosure", "functional safety", "real-robot certification"),
        ),
        (
            "configs/deploy/tron2_deploy.client.example.yaml",
            ("source disclosure", "functional safety", "real-robot certification"),
        ),
        ("configs/deploy/tron2_deploy.server.example_CN.yaml", ("源码公开", "功能安全", "真机认证")),
        ("configs/deploy/tron2_deploy.client.example_CN.yaml", ("源码公开", "功能安全", "真机认证")),
    ],
)
def test_source_disclosure_is_not_described_as_safety_certification(relative_path: str, terms: tuple[str, ...]):
    _assert_terms(_read_repository_file(relative_path), terms)


def test_repository_has_no_submodule_metadata_or_gitlinks():
    assert not (REPOSITORY_ROOT / ".gitmodules").exists()

    index_entries = subprocess.run(
        ["git", "ls-files", "--stage"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not [entry for entry in index_entries if entry.startswith("160000 ")]


def test_tracked_markdown_has_no_stale_submodule_instructions():
    tracked_docs = subprocess.run(
        ["git", "ls-files", "--", "*.md"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    for relative_path in tracked_docs:
        if not (REPOSITORY_ROOT / relative_path).exists():
            continue
        normalized = _read_repository_file(relative_path).lower()
        assert "git submodule" not in normalized
        assert "recurse-submodules" not in normalized


@pytest.mark.parametrize(("relative_path", "source_url", "commit", "destination"), EXTERNAL_EXAMPLE_SOURCES)
def test_external_example_sources_are_pinned(relative_path: str, source_url: str, commit: str, destination: str):
    content = _read_repository_file(relative_path)

    assert source_url in content
    assert commit in content
    _assert_terms(
        content,
        (destination, "external checkout", "not included", "license", "git clone", "checkout --detach"),
    )


def test_external_example_checkouts_are_ignored_and_disclosed():
    assert "/third_party/" in _read_repository_file(".gitignore").splitlines()
    _assert_terms(
        _read_repository_file("NOTICE"),
        ("external source checkouts", "not included", "source snapshot", "third_party/aloha", "third_party/libero"),
    )
    _assert_terms(_read_repository_file("CHANGELOG.md"), ("external example dependencies",))


def test_upstream_license_copies_match_the_approved_texts():
    for relative_path, expected_hash in APPROVED_LICENSE_HASHES.items():
        content = (REPOSITORY_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash


def test_third_party_inventory_records_sources_scope_and_license_files():
    content = _read_repository_file("THIRD_PARTY_NOTICES.md")

    _assert_terms(
        content,
        (
            "OpenPI",
            "working baseline + exact origin unknown",
            OPENPI_WORKING_BASELINE,
            "Big Vision",
            "Google Vision Transformer",
            "Hugging Face Transformers",
            "msgpack-numpy",
            "20c5e5b",
            "ACT / ALOHA copied files",
            "742c753",
            "DROID copied section",
            "e9254e3",
            "robosuite snippet",
            "eafb81f",
            "Kinetix RTC",
            "9296f31",
            "LeRobot RTC",
            "e40b58a",
            "external ALOHA checkout",
            "external LIBERO checkout",
            "not included",
            "current source snapshot",
        ),
    )
    for field in ("Local path", "Classification", "Runtime use", "Distributed in", "Modified", "License"):
        assert field in content
    for relative_path in APPROVED_LICENSE_HASHES:
        assert relative_path in content


def test_modifications_record_approved_baseline_and_initial_comparison():
    content = _read_repository_file("MODIFICATIONS.md")

    _assert_terms(
        content,
        (
            "working baseline + exact origin unknown",
            OPENPI_WORKING_BASELINE,
            "121 common blobs",
            "16 common paths modified",
            "18 local additions",
            "initial comparison",
            "later governance",
        ),
    )
    for relative_path in (
        ".dockerignore",
        ".gitignore",
        "CONTRIBUTING.md",
        "README.md",
        "examples/droid/main.py",
        "packages/openpi-client/pyproject.toml",
        "packages/openpi-client/src/openpi_client/websocket_client_policy.py",
        "pyproject.toml",
        "scripts/serve_policy.py",
        "src/openpi/models/pi0.py",
        "src/openpi/models/pi0_config.py",
        "src/openpi/policies/policy.py",
        "src/openpi/serving/websocket_policy_server.py",
        "src/openpi/training/config.py",
        "src/openpi/training/data_loader.py",
        "uv.lock",
    ):
        assert f"`{relative_path}`" in content


def test_gemma_terms_are_scoped_to_external_model_assets_not_source_code():
    gemma_hash = hashlib.sha256((REPOSITORY_ROOT / "LICENSE_GEMMA.txt").read_bytes()).hexdigest()
    assert gemma_hash == "3e2c24001f9ef57bf7ec959a3658fbb49cdad113cdf394c264da9d16f9bdd132"

    english = "\n".join(_read_repository_file(path) for path in ("README.md", "NOTICE"))
    _assert_terms(
        english,
        (
            "file-level source licenses",
            "upstream-carried model asset terms",
            "no Gemma or PaliGemma weights",
            "model derivatives",
            "applicable Gemma Terms",
            "relicense",
            "restrictions",
            "Apache source",
            "Hosted Service",
            "re-review",
        ),
    )
    assert "Gemma-related components are also subject" not in english
    assert "Gemma-related use is subject" not in english

    chinese = _read_repository_file("README_CN.md")
    _assert_terms(
        chinese,
        (
            "文件级源码许可证",
            "上游模型资产条款材料",
            "不包含 Gemma 或 PaliGemma 权重",
            "模型衍生物",
            "适用的 Gemma Terms",
            "不会重新许可",
            "Apache 源码",
            "Hosted Service",
            "重新评审",
        ),
    )
    assert "Gemma 相关组件还受" not in chinese


def test_changelog_records_provenance_docs_without_a_legal_approval_claim():
    content = _read_repository_file("CHANGELOG.md")

    _assert_terms(content, ("license and provenance documentation",))
    assert "legal approval" not in content.lower()


def _tracked_text_contains_concrete_home_path() -> bool:
    tracked_paths = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    home_prefix = b"/" + b"home/"
    home_path = re.compile(re.escape(home_prefix) + rb"([^/\s]+)/")

    for encoded_path in tracked_paths:
        if not encoded_path:
            continue
        path = REPOSITORY_ROOT / encoded_path.decode()
        if not path.exists():
            continue
        content = path.read_bytes()
        if b"\0" in content:
            continue
        for match in home_path.finditer(content):
            username = match.group(1)
            if any(marker in username for marker in (b"<", b">", b"$", b"{", b"}")):
                continue
            return True
    return False


def test_tracked_text_has_no_concrete_personal_home_paths():
    assert not _tracked_text_contains_concrete_home_path()


def test_act_provenance_distinguishes_the_final_comment_only_modification():
    inventory = _read_repository_file("THIRD_PARTY_NOTICES.md")
    act_section = inventory.split("## ACT / ALOHA copied files", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]

    assert "Modified: identical to the OpenPI working baseline" not in act_section
    _assert_terms(
        act_section,
        (
            "constants.py",
            "real_env.py",
            "remain identical",
            "robot_utils.py",
            "locally modified",
            "non-functional personal-path comment",
            "no runtime behavior change",
            "exact component-origin diff remains unknown",
        ),
    )

    _assert_terms(
        _read_repository_file("MODIFICATIONS.md"),
        (
            "final snapshot cleanup",
            "examples/aloha_real/robot_utils.py",
            "non-functional personal-path comment",
            "no runtime behavior change",
        ),
    )
