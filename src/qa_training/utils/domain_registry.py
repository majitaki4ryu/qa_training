from qa_training.adapter.repo_input_data import RepoInputData
from qa_training.adapter.repo_model import RepoModel
from qa_training.adapter.repo_output_data import RepoOutputData
from qa_training.usecase.usecase_create_model import UsecaseCreateModel
from qa_training.usecase.usecase_judge_survival import UsecaseJudgeSurvival
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


class DomainRegistry:
    """CongiManagerの返り値を元に抽象クラスの依存解決をする."""

    def __init__(self, config_manager: ConfigManager) -> None:
        assert type(config_manager) is ConfigManager
        self._config_manager = config_manager

    # Usecase
    def usecase_judge_survival(self) -> IF_UsecaseJudgeSurvival:
        (
            classname,
            dict_params,
        ) = self._config_manager.params_for_usecase_judge_survival()

        if classname == UsecaseJudgeSurvival.__name__:
            repo_model = self.repo_model()
            repo_input_data = self.repo_input_data()
            repo_output_data = self.repo_output_data()
            return UsecaseJudgeSurvival(
                repo_model=repo_model,
                repo_input_data=repo_input_data,
                repo_output_data=repo_output_data,
                **dict_params,
            )
        else:
            raise ValueError

    def usecase_create_model(self) -> IF_UsecaseCreateModel:
        (
            classname,
            dict_params,
        ) = self._config_manager.params_for_usecase_create_model()

        if classname == UsecaseCreateModel.__name__:
            repo_model = self.repo_model()
            repo_input_data = self.repo_input_data()
            return UsecaseCreateModel(
                repo_model=repo_model,
                repo_input_data=repo_input_data,
                **dict_params,
            )
        else:
            raise ValueError

    # Repository
    def repo_model(self) -> IF_RepoModel:
        classname, dict_params = self._config_manager.params_for_repo_model()
        if classname == RepoModel.__name__:
            return RepoModel(**dict_params)
        else:
            raise ValueError

    def repo_input_data(self) -> IF_RepoInputData:
        classname, dict_params = self._config_manager.params_for_repo_input_data()
        if classname == RepoInputData.__name__:
            return RepoInputData(**dict_params)
        else:
            raise ValueError

    def repo_output_data(self) -> IF_RepoOutputData:
        classname, dict_params = self._config_manager.params_for_repo_output_data()
        if classname == RepoOutputData.__name__:
            return RepoOutputData(**dict_params)
        else:
            raise ValueError
