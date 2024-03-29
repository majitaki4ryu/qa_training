from qa_training.utils.boundary.repo.if_repo_input_data import IF_RepoInputData
from qa_training.utils.boundary.repo.if_repo_model import IF_RepoModel
from qa_training.utils.boundary.repo.if_repo_output_data import IF_RepoOutputData
from qa_training.utils.boundary.usecase.if_usecase_create_model import (
    IF_UsecaseCreateModel,
)
from qa_training.utils.boundary.usecase.if_usecase_judge_survival import (
    IF_UsecaseJudgeSurvival,
)
from qa_training.utils.config_manager import ConfigManager
from qa_training.utils.domain_registry import DomainRegistry


def test_domain_registry_initialize(config_manager: ConfigManager) -> None:
    """DomainRegistryの初期化できるか."""
    domain_registry = DomainRegistry(config_manager)
    assert isinstance(domain_registry, DomainRegistry)


# Usecase
def test_usecase(domain_registry: DomainRegistry) -> None:
    """DomainRegistryがUsecaseを正常に生成できるか."""
    usecase_judge_survival = domain_registry.usecase_judge_survival()
    assert isinstance(usecase_judge_survival, IF_UsecaseJudgeSurvival)

    usecase_create_model = domain_registry.usecase_create_model()
    assert isinstance(usecase_create_model, IF_UsecaseCreateModel)


# repo
def test_repo(domain_registry: DomainRegistry) -> None:
    """DomainRegistryがrepoを正常に生成できるか."""
    repo_model = domain_registry.repo_model()
    assert isinstance(repo_model, IF_RepoModel)

    repo_input_data = domain_registry.repo_input_data()
    assert isinstance(repo_input_data, IF_RepoInputData)

    repo_input_data = domain_registry.repo_output_data()
    assert isinstance(repo_input_data, IF_RepoOutputData)
