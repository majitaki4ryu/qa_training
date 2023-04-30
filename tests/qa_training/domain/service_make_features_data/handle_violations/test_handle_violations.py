import pandas as pd
import pytest
from qa_training.domain.service_make_features import ServiceMakeFeatures
from qa_training.utils.my_assert_frame_equal import MyAssert

test_data_common_path = (
    "tests/qa_training/domain/service_make_features_data/handle_violations"
)


@pytest.fixture
def fixture_handle_violations_in_pclass():
    service_make_features = ServiceMakeFeatures()

    dir_path = "handle_violations_in_pclass"

    df_customer_info = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_customer_info.csv"
    )
    df_obeyed_expected = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_obeyed_expected.csv"
    )

    return (service_make_features, df_customer_info, df_obeyed_expected)


def test_handle_violations_in_pclass(
    fixture_handle_violations_in_pclass: tuple[
        ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
    ]
):
    """制約違反処理, pclassが1,2,3以外の行が削除されること"""

    (
        service_make_features,
        df_customer_info,
        df_obeyed_expected,
    ) = fixture_handle_violations_in_pclass

    df_obeyed = service_make_features._handle_violations(df_customer_info)

    MyAssert().assert_df(df_obeyed, df_obeyed_expected)


@pytest.fixture
def fixture_handle_violations_in_sex():
    service_make_features = ServiceMakeFeatures()

    dir_path = "handle_violations_in_sex"

    df_customer_info = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_customer_info.csv"
    )
    df_obeyed_expected = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_obeyed_expected.csv"
    )

    return (service_make_features, df_customer_info, df_obeyed_expected)


def test_handle_violations_in_sex(
    fixture_handle_violations_in_sex: tuple[
        ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
    ]
):
    """制約違反処理, sexがmale, female以外の行が削除されること"""

    (
        service_make_features,
        df_customer_info,
        df_obeyed_expected,
    ) = fixture_handle_violations_in_sex

    df_obeyed = service_make_features._handle_violations(df_customer_info)

    MyAssert().assert_df(df_obeyed, df_obeyed_expected)


@pytest.fixture
def fixture_handle_violations_in_age():
    service_make_features = ServiceMakeFeatures()

    dir_path = "handle_violations_in_age"

    df_customer_info = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_customer_info.csv"
    )
    df_obeyed_expected = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_obeyed_expected.csv"
    )

    return (service_make_features, df_customer_info, df_obeyed_expected)


def test_handle_violations_in_age(
    fixture_handle_violations_in_age: tuple[
        ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
    ]
):
    """制約違反処理, ageが0以上130以下の整数であること"""

    (
        service_make_features,
        df_customer_info,
        df_obeyed_expected,
    ) = fixture_handle_violations_in_age

    df_obeyed = service_make_features._handle_violations(df_customer_info)

    MyAssert().assert_df(df_obeyed, df_obeyed_expected)


@pytest.fixture
def fixture_handle_violations_in_fare():
    service_make_features = ServiceMakeFeatures()

    dir_path = "handle_violations_in_fare"

    df_customer_info = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_customer_info.csv"
    )
    df_obeyed_expected = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_obeyed_expected.csv"
    )

    return (service_make_features, df_customer_info, df_obeyed_expected)


def test_handle_violations_in_fare(
    fixture_handle_violations_in_fare: tuple[
        ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
    ]
):
    """制約違反処理, fareが0以上1000以下の少数であること"""

    (
        service_make_features,
        df_customer_info,
        df_obeyed_expected,
    ) = fixture_handle_violations_in_fare

    df_obeyed = service_make_features._handle_violations(df_customer_info)

    MyAssert().assert_df(df_obeyed, df_obeyed_expected)


@pytest.fixture
def fixture_handle_violations_in_embarked():
    service_make_features = ServiceMakeFeatures()

    dir_path = "handle_violations_in_embarked"

    df_customer_info = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_customer_info.csv"
    )
    df_obeyed_expected = pd.read_csv(
        f"{test_data_common_path}/{dir_path}/df_obeyed_expected.csv"
    )

    return (service_make_features, df_customer_info, df_obeyed_expected)


def test_handle_violations_in_embarked(
    fixture_handle_violations_in_embarked: tuple[
        ServiceMakeFeatures, pd.DataFrame, pd.DataFrame
    ]
):
    """制約違反処理, embarkedがC, Q, S以外の行が削除されること"""

    (
        service_make_features,
        df_customer_info,
        df_obeyed_expected,
    ) = fixture_handle_violations_in_embarked

    df_obeyed = service_make_features._handle_violations(df_customer_info)

    MyAssert().assert_df(df_obeyed, df_obeyed_expected)
