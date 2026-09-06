import os

import yaml

REQUIRED_OPSASSIST_FOLDERS = [
    "api",
    "agents",
    "rag",
    "tools",
    "connectors",
    "evals",
    "guardrails",
    "observability",
    "prompts",
]

REQUIRED_TOP_LEVEL_PATHS = [
    "docs/architecture.md",
    "configs/opsassist.dev.yaml",
    ".env.example",
]


def test_all_required_opsassist_folders_exist():
    missing = [
        folder
        for folder in REQUIRED_OPSASSIST_FOLDERS
        if not os.path.isdir(os.path.join("src", "opsassist", folder))
    ]
    assert not missing, f"Missing src/opsassist/ folders: {missing}"


def test_all_required_top_level_files_exist():
    missing = [p for p in REQUIRED_TOP_LEVEL_PATHS if not os.path.isfile(p)]
    assert not missing, f"Missing required files: {missing}"


def test_opsassist_config_is_valid_yaml_with_required_sections():
    with open("configs/opsassist.dev.yaml") as f:
        config = yaml.safe_load(f)

    for section in ("application", "llm", "retrieval", "guardrails", "observability"):
        assert section in config, f"Missing config section: {section}"


def test_llm_provider_is_one_of_the_supported_backends():
    with open("configs/opsassist.dev.yaml") as f:
        config = yaml.safe_load(f)

    assert config["llm"]["provider"] in {"openai", "azure", "aws"}


def test_env_example_declares_provider_credentials_without_real_values():
    with open(".env.example") as f:
        content = f.read()

    expected_vars = [
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AWS_BEDROCK_REGION",
    ]
    for var in expected_vars:
        assert var in content, f"{var} should be declared in .env.example"

    # Credential placeholders must stay empty -- a real key accidentally
    # committed here would be a serious leak, not just a style issue.
    for line in content.splitlines():
        if "API_KEY" in line or "SECRET" in line:
            key, _, value = line.partition("=")
            assert (
                value.strip() == ""
            ), f"{key} appears to have a real value in .env.example"
