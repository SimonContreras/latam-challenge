"""DelayModel class module, implements methods to preprocess data for training 
and serving, fit model and make predictions"""
import logging
import os
import pickle
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
LOG_FMT = "%(levelname)s:   %(asctime)s - %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger(__name__)

DAY_MONTH_FMT = "%d-%b"
HOUR_MIN_FMT = "%H:%M"
FULL_DATE_FMT = "%Y-%m-%d %H:%M:%S"


DATE_O = "Fecha-O"
DATE_I = "Fecha-I"
PERIOD_DAY = "period_day"
HIGH_SEASON = "high_season"
MIN_DIFF = "min_diff"
DELAY = "delay"
OPERA = "OPERA"
FLIGHT_TYPE = "TIPOVUELO"
MONTH = "MES"


class DayTime(Enum):
    """Enum class to map DayTimes constants"""

    MORNING = "mañana"
    AFTERNOON = "tarde"
    NIGHT = "noche"


class DelayModel:
    """Delay Model main class, provides methods to preprocess the data,
    fit the model and make predictions.
    """

    def __init__(self):
        self.model_path = Path(os.getcwd(), "delay_model.pkl")
        self._model = self._load_model()
        self._unfiltered_features: pd.DataFrame = None
        self._target_column: str = None
        self.top_10_features: List[str] = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        self.dataset_cols: List[str] = [
            "Fecha-I",
            "Vlo-I",
            "Ori-I",
            "Des-I",
            "Emp-I",
            "Fecha-O",
            "Vlo-O",
            "Ori-O",
            "Des-O",
            "Emp-O",
            "DIA",
            "MES",
            "AÑO",
            "DIANOM",
            "TIPOVUELO",
            "OPERA",
            "SIGLAORI",
            "SIGLADES",
        ]
        self.derived_cols: List[str] = [
            "period_day",
            "high_season",
            "min_diff",
            "delay",
        ]

    @staticmethod
    def _is_high_season(date: str) -> int:
        """Classify if a date is a high season or not.

        Args:
            date (str): date to classify.

        Returns:
            int: 1 if the date is high season, otherwise 0.
        """
        year_date = int(date.split("-")[0])
        date = datetime.strptime(date, FULL_DATE_FMT)
        range1_min = datetime.strptime("15-Dec", DAY_MONTH_FMT).replace(year=year_date)
        range1_max = datetime.strptime("31-Dec", DAY_MONTH_FMT).replace(year=year_date)
        range2_min = datetime.strptime("1-Jan", DAY_MONTH_FMT).replace(year=year_date)
        range2_max = datetime.strptime("3-Mar", DAY_MONTH_FMT).replace(year=year_date)
        range3_min = datetime.strptime("15-Jul", DAY_MONTH_FMT).replace(year=year_date)
        range3_max = datetime.strptime("31-Jul", DAY_MONTH_FMT).replace(year=year_date)
        range4_min = datetime.strptime("11-Sep", DAY_MONTH_FMT).replace(year=year_date)
        range4_max = datetime.strptime("30-Sep", DAY_MONTH_FMT).replace(year=year_date)

        if (
            (date >= range1_min and date <= range1_max)
            or (date >= range2_min and date <= range2_max)
            or (date >= range3_min and date <= range3_max)
            or (date >= range4_min and date <= range4_max)
        ):
            return 1
        else:
            return 0

    @staticmethod
    def _get_period_day(date: str) -> str:
        """Based on time intervals, map a date to a interval
        defined on DayTime enum class.

        Args:
            date (str): date to map.

        Returns:
            str: date mapped to a DayTime value.
        """
        date_time = datetime.strptime(date, FULL_DATE_FMT).time()
        morning_min = datetime.strptime("05:00", HOUR_MIN_FMT).time()
        morning_max = datetime.strptime("11:59", HOUR_MIN_FMT).time()
        afternoon_min = datetime.strptime("12:00", HOUR_MIN_FMT).time()
        afternoon_max = datetime.strptime("18:59", HOUR_MIN_FMT).time()
        evening_min = datetime.strptime("19:00", HOUR_MIN_FMT).time()
        evening_max = datetime.strptime("23:59", HOUR_MIN_FMT).time()
        night_min = datetime.strptime("00:00", HOUR_MIN_FMT).time()
        night_max = datetime.strptime("4:59", HOUR_MIN_FMT).time()

        if date_time > morning_min and date_time < morning_max:
            return DayTime.MORNING.value
        elif date_time > afternoon_min and date_time < afternoon_max:
            return DayTime.AFTERNOON.value
        elif (date_time > evening_min and date_time < evening_max) or (
            date_time > night_min and date_time < night_max
        ):
            return DayTime.NIGHT.value

    @staticmethod
    def _get_min_diff(data: pd.DataFrame) -> int:
        """Generates MIN_DIFF value for a Dataframe.

        Args:
            data (pd.DataFrame): Dataframe to use.

        Returns:
            int: min_diff value.
        """
        date_o = datetime.strptime(data[DATE_O], FULL_DATE_FMT)
        date_i = datetime.strptime(data[DATE_I], FULL_DATE_FMT)
        min_diff = ((date_o - date_i).total_seconds()) / 60
        return min_diff

    def _all_columns_required_exists(self, data: pd.DataFrame) -> bool:
        """Check that all minimal columns required to preprocess and fit the
        model exists on the current Dataframe.

        Args:
            data (pd.DataFrame): Dataframe to evaluate.

        Returns:
            bool: True if all columns exists, otherwise False.
        """
        return all(ele in list(self.dataset_cols) for ele in list(data.columns))

    def _is_valid_target(self, target_column: str) -> bool:
        """Check if a column is valid for the current model.

        Args:
            target_column (str): column to check.

        Returns:
            bool: True if the column is valid, otherwise False.
        """
        return target_column in [*self.dataset_cols, *self.derived_cols]

    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for columns OPERA, FLIGHT_TYPE and MONTH
        values.
        Args:
            data (pd.DataFrame): Dataset to generate new features.

        Returns:
            pd.DataFrame: New Dataframe with features.
        """
        features = pd.concat(
            [
                pd.get_dummies(data[OPERA], prefix=OPERA),
                pd.get_dummies(data[FLIGHT_TYPE], prefix=FLIGHT_TYPE),
                pd.get_dummies(data[MONTH], prefix=MONTH),
            ],
            axis=1,
        )
        return features.copy(True)

    def _generate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Created derived feautures PERIOD_DAY, HIGH_SEASON, MIN_DIFF and DELAY
        Args:
            data (pd.DataFrame): Dataset used to generates new features.

        Returns:
            pd.DataFrame: Dataframe with new features.
        """
        threshold_in_minutes = 15
        data[PERIOD_DAY] = data[DATE_I].apply(self._get_period_day)
        data[HIGH_SEASON] = data[DATE_I].apply(self._is_high_season)
        data[MIN_DIFF] = data.apply(self._get_min_diff, axis=1)
        data[DELAY] = np.where(data[MIN_DIFF] > threshold_in_minutes, 1, 0)
        return data.copy(True)

    def _serving_feature_to_top_10_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Converts DataFrame of data received from an API call to the
        format that use the model with the top 10 features.
        Args:
            data (pd.DataFrame): data payload from API call formatted as Dataframe.

        Returns:
            pd.DataFrame: Dataframe converted with top 10 features format.
        """
        default_top_10_features = pd.DataFrame(
            0, index=np.arange(data.shape[0]), columns=self.top_10_features
        )
        for col in data.columns:
            if col in default_top_10_features.columns:
                default_top_10_features[col] = default_top_10_features[col] | data[col]
        return default_top_10_features

    def _save_as_pickle(self, model: LogisticRegression) -> Path:
        """Save a model as pickle file on self.model_path.
        Args:
            model (LogisticRegression): Model to be pickled.

        Returns:
            Path: Path wher the model was saved.
        """
        with open(self.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    def _load_model(self) -> Union[LogisticRegression, None]:
        """If a pickled model exists on self.model_path it's loaded
        on the class attribute self._model.

        Returns:
            Union[LogisticRegression, None]: Model loaded,
                Otherwise None.
        """
        loaded_model = None
        if self.model_path.is_file():
            with open(self.model_path, "rb") as model_file:
                loaded_model = pickle.load(model_file)
        return loaded_model

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        if self._all_columns_required_exists(data) and self._is_valid_target(
            target_column
        ):
            df_with_derived_feats = self._generate_derived_features(data)
            features = self._generate_features(df_with_derived_feats)
            self.unfiltered_features = features.copy(True)
            self._target_column = target_column
            return features[self.top_10_features], data[[target_column]]
        else:
            serving_features = self._generate_features(data)
            top_10_features_df = self._serving_feature_to_top_10_format(
                serving_features
            )
            return top_10_features_df[self.top_10_features]

    def _returns_scale_values(self, target: pd.DataFrame) -> Tuple[int, int]:
        """Creates n_y0 and n_y1 values for balance the model
        for future fit using top_10_features.

        Args:
            target (pd.DataFrame): target column

        Returns:
            Tuple[int, int]: n_y0 and n_y1 values.
        """
        _, _, y_train, _ = train_test_split(
            self.unfiltered_features, target, test_size=0.33, random_state=42
        )
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        return n_y0, n_y1

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        n_y0, n_y1 = self._returns_scale_values(target[self._target_column].copy(True))
        x_train, _, y_train, _ = train_test_split(
            features, target, test_size=0.33, random_state=42
        )
        reg_model = LogisticRegression(
            class_weight={1: n_y0 / len(y_train), 0: n_y1 / len(y_train)}
        )
        reg_model.fit(x_train, np.ravel(y_train))
        self._model = reg_model
        self._save_as_pickle(reg_model)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        preds = self._model.predict(features)
        return preds.tolist()
