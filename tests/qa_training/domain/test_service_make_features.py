import numpy as np
import pandas as pd
import pytest
from qa_training.domain.service_make_features import ServiceMakeFeatures
from qa_training.utils.my_assert_frame_equal import MyAssert


@pytest.fixture
def fixture_run():
    service_make_features = ServiceMakeFeatures()

    df_customer_info = pd.read_csv(
        "tests/common_data/df_customer_info.csv",
    )
    df_id_expected = pd.read_csv(
        "tests/common_data/df_id.csv",
    )
    df_X_expected = pd.read_csv(
        "tests/common_data/df_X.csv",
    )
    df_y_expected = pd.read_csv(
        "tests/common_data/df_y.csv",
    )
    return (
        service_make_features,
        df_customer_info,
        df_id_expected,
        df_X_expected,
        df_y_expected,
    )


def test_run(
    fixture_run: tuple[
        ServiceMakeFeatures, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]
):
    (
        service_make_features,
        df_customer_info,
        df_id_expected,
        df_X_expected,
        df_y_expected,
    ) = fixture_run

    df_id, df_X, df_y = service_make_features.run(df_customer_info)

    MyAssert().assert_df(df_id, df_id_expected)
    MyAssert().assert_df(df_X, df_X_expected)
    MyAssert().assert_df(df_y, df_y_expected)

def test_handle_missing_values(
    fixture_run: tuple[
        ServiceMakeFeatures, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]
):
    (
        service_make_features,
        df_customer_info,
        df_id_expected,
        df_X_expected,
        df_y_expected,
    ) = fixture_run
    test_df = pd.DataFrame({
        'Sex':[np.nan, 'male', 'female', np.nan, 'female'],
        'Age':[21, 15, np.nan, np.nan, 33],
        'Embarked':['C', np.nan, 'Q', 'Q', 'S'],
        'Pclass':[1, np.nan, 1, 1, 3],
        'Cabin':['C123', 'C85', 'B42', 'C33', 'S33'],
        'Name':['Alen', 'Bob', 'aaa', 'Cas', np.nan],
        'Survival':['Alen', 'Bob', 'aaa', 'Cas', np.nan],
        'Sibsp':['Alen', 'Bob', 'aaa', 'Cas', np.nan],
        'Parch':['Alen', 'Bob', 'aaa', 'Cas', np.nan],
        'Ticket':['Alen', 'Bob', 'aaa', 'Cas', np.nan],
        'Fare':['Alen', 'Bob', 'aaa', 'Cas', np.nan],
    })
    test_expected_df = pd.DataFrame({
        'Sex':['male', 'male', 'female', 'male'],
        'Age':[21, 15, 20, 20],
        'Embarked':['C', 'S', 'Q', 'Q'],
        'Pclass':[1, 2, 1, 1],
        'Cabin':['C123', 'C85', 'B42', 'C33'],
        'Name':['Alen', 'Bob', 'aaa', 'Cas'],
        'Survival':['Alen', 'Bob', 'aaa', 'Cas'],
        'Sibsp':['Alen', 'Bob', 'aaa', 'Cas'],
        'Parch':['Alen', 'Bob', 'aaa', 'Cas'],
        'Ticket':['Alen', 'Bob', 'aaa', 'Cas'],
        'Fare':['Alen', 'Bob', 'aaa', 'Cas'],
    })
    out_put = service_make_features._handle_missing_values(test_df)
    MyAssert().assert_df(test_expected_df, out_put)



def test_handle_violations(
    fixture_run: tuple[
        ServiceMakeFeatures, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]
):
    (
        service_make_features,
        df_customer_info,
        df_id_expected,
        df_X_expected,
        df_y_expected,
    ) = fixture_run
    test_df = pd.DataFrame({
        'Sex':['male', 'male', 'female', 'male'],
        'Age':[21, 15, 20, 20],
        'Embarked':['C', 'S', 'Q', 'Q'],
        'Pclass':[1, 2, 1, 1],
        'Cabin':['C123', 'C85', 'B42', 'C33'],
        'Name':['Alen', 'Bob', 'aaa', 'Cas'],
        'Survival':['Alen', 'Bob', 'aaa', 'Cas'],
        'Sibsp':['Alen', 'Bob', 'aaa', 'Cas'],
        'Parch':['Alen', 'Bob', 'aaa', 'Cas'],
        'Ticket':['Alen', 'Bob', 'aaa', 'Cas'],
        'Fare':['Alen', 'Bob', 'aaa', 'Cas'],
    })
    test_expected_df = pd.DataFrame({
        'Sex':['male', 'male', 'female', 'male'],
        'Age':[21, 15, 20, 20],
        'Embarked':['C', 'S', 'Q', 'Q'],
        'Pclass':[1, 2, 1, 1],
        'Cabin':['C123', 'C85', 'B42', 'C33'],
        'Name':['Alen', 'Bob', 'aaa', 'Cas'],
        'Survival':[0, 1, 1, 1],
        'Sibsp':['Alen', 'Bob', 'aaa', 'Cas'],
        'Parch':['Alen', 'Bob', 'aaa', 'Cas'],
        'Ticket':['Alen', 'Bob', 'aaa', 'Cas'],
        'Fare':['Alen', 'Bob', 'aaa', 'Cas'],
    })
    out_put = service_make_features._handle_violations(test_df)
    MyAssert().assert_df(test_expected_df, out_put)
