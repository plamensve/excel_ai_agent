import pandas as pd


df = pd.read_excel("../data/data.xlsx")
client_cards = pd.read_excel("../data/cards.xlsx")


class AgentBaseFunctions:

    def __init__(self, dataset, cards):
        self.dataset = dataset.copy()
        self.cards = cards.copy()

        # Normalize card numbers as strings
        if "Number" in self.dataset.columns:
            self.dataset["Number"] = (
                self.dataset["Number"]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )

        if "Number" in self.cards.columns:
            self.cards["Number"] = (
                self.cards["Number"]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )

        # Create one enriched dataset that can be used internally
        self.enriched_dataset = self._create_enriched_dataset()

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    def _create_enriched_dataset(self):
        """
        Creates a dataset containing transaction data together
        with company and vehicle information.
        """

        if "Number" not in self.dataset.columns:
            return self.dataset.copy()

        if "Number" not in self.cards.columns:
            return self.dataset.copy()

        available_columns = [
            column
            for column in ["Number", "Company", "Name"]
            if column in self.cards.columns
        ]

        cards = (
            self.cards[available_columns]
            .drop_duplicates(subset=["Number"])
        )

        dataset = self.dataset.merge(
            cards,
            on="Number",
            how="left"
        )

        return dataset

    def _validate_column(self, column):
        if column not in self.enriched_dataset.columns:
            return False

        return True

    def _apply_filters(self, dataset, filters):
        """
        Applies a list of filters.

        Supported operators:
        eq, neq, contains, not_contains,
        gt, gte, lt, lte, in
        """

        if not filters:
            return dataset

        filtered = dataset.copy()

        for condition in filters:

            column = condition.get("column")
            operator = condition.get("operator", "eq")
            value = condition.get("value")

            if column not in filtered.columns:
                continue

            series = filtered[column]

            if operator == "eq":
                filtered = filtered[
                    series.astype(str).str.lower()
                    == str(value).lower()
                ]

            elif operator == "neq":
                filtered = filtered[
                    series.astype(str).str.lower()
                    != str(value).lower()
                ]

            elif operator == "contains":
                filtered = filtered[
                    series.astype(str).str.contains(
                        str(value),
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

            elif operator == "not_contains":
                filtered = filtered[
                    ~series.astype(str).str.contains(
                        str(value),
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

            elif operator == "gt":
                numeric = pd.to_numeric(series, errors="coerce")
                filtered = filtered[numeric > float(value)]

            elif operator == "gte":
                numeric = pd.to_numeric(series, errors="coerce")
                filtered = filtered[numeric >= float(value)]

            elif operator == "lt":
                numeric = pd.to_numeric(series, errors="coerce")
                filtered = filtered[numeric < float(value)]

            elif operator == "lte":
                numeric = pd.to_numeric(series, errors="coerce")
                filtered = filtered[numeric <= float(value)]

            elif operator == "in":
                values = [str(v).lower() for v in value]

                filtered = filtered[
                    series.astype(str).str.lower().isin(values)
                ]

        return filtered

    # =========================================================
    # DATASET INFORMATION
    # =========================================================

    def get_dataset_shape(self):
        return {
            "rows": self.enriched_dataset.shape[0],
            "columns": self.enriched_dataset.shape[1]
        }

    def get_columns(self):
        """
        Returns columns together with their data types.
        """

        return [
            {
                "column": column,
                "dtype": str(self.enriched_dataset[column].dtype)
            }
            for column in self.enriched_dataset.columns
        ]

    def get_missing_values(self):
        return (
            self.enriched_dataset
            .isna()
            .sum()
            .to_dict()
        )

    def get_column_values(self, column, limit=50):
        """
        Returns unique values from a column.

        Useful when the LLM needs to discover exact values such as
        DIESEL EKONOMY, EKO RACING 100, etc.
        """

        if not self._validate_column(column):
            return {
                "error": f"Column '{column}' does not exist."
            }

        values = (
            self.enriched_dataset[column]
            .dropna()
            .astype(str)
            .value_counts()
            .head(limit)
        )

        return [
            {
                "value": index,
                "count": int(count)
            }
            for index, count in values.items()
        ]

    # =========================================================
    # BASIC STATISTICS
    # =========================================================

    def calculate_mean(self, column, filters=None):

        if not self._validate_column(column):
            return {
                "error": f"Column '{column}' does not exist."
            }

        dataset = self._apply_filters(
            self.enriched_dataset,
            filters
        )

        values = pd.to_numeric(
            dataset[column],
            errors="coerce"
        )

        return {
            "column": column,
            "mean": float(values.mean()),
            "rows_used": int(values.notna().sum())
        }

    def calculate_sum(self, column, filters=None):

        if not self._validate_column(column):
            return {
                "error": f"Column '{column}' does not exist."
            }

        dataset = self._apply_filters(
            self.enriched_dataset,
            filters
        )

        values = pd.to_numeric(
            dataset[column],
            errors="coerce"
        )

        return {
            "column": column,
            "sum": float(values.sum()),
            "rows_used": int(values.notna().sum())
        }

    def calculate_statistics(self, column, filters=None):
        """
        Returns common descriptive statistics.
        """

        if not self._validate_column(column):
            return {
                "error": f"Column '{column}' does not exist."
            }

        dataset = self._apply_filters(
            self.enriched_dataset,
            filters
        )

        values = pd.to_numeric(
            dataset[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            return {
                "error": "No numeric values found."
            }

        return {
            "column": column,
            "count": int(values.count()),
            "sum": float(values.sum()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std())
        }

    # =========================================================
    # UNIVERSAL GROUP BY
    # =========================================================

    def group_and_aggregate(
        self,
        group_by,
        value_column,
        aggregation="sum",
        filters=None,
        sort="desc",
        limit=20
    ):
        """
        Universal analytical function.

        Examples:

        Which company consumed the most diesel?
        group_by = Company
        value_column = Bill.qty
        aggregation = sum

        Which card has the highest consumption?
        group_by = Number
        value_column = Bill.qty
        """

        if group_by not in self.enriched_dataset.columns:
            return {
                "error": f"Column '{group_by}' does not exist."
            }

        if value_column not in self.enriched_dataset.columns:
            return {
                "error": f"Column '{value_column}' does not exist."
            }

        allowed_aggregations = {
            "sum",
            "mean",
            "count",
            "min",
            "max",
            "median"
        }

        if aggregation not in allowed_aggregations:
            return {
                "error": (
                    f"Unsupported aggregation '{aggregation}'. "
                    f"Allowed: {sorted(allowed_aggregations)}"
                )
            }

        dataset = self._apply_filters(
            self.enriched_dataset,
            filters
        )

        if dataset.empty:
            return []

        dataset = dataset.copy()

        if aggregation != "count":
            dataset[value_column] = pd.to_numeric(
                dataset[value_column],
                errors="coerce"
            )

        grouped = (
            dataset
            .groupby(group_by, dropna=False)[value_column]
            .agg(aggregation)
            .reset_index()
        )

        result_column = f"{aggregation}_{value_column}"

        grouped = grouped.rename(
            columns={
                value_column: result_column
            }
        )

        ascending = sort == "asc"

        grouped = (
            grouped
            .sort_values(
                result_column,
                ascending=ascending
            )
            .head(limit)
        )

        return grouped.to_dict(orient="records")

    # =========================================================
    # MULTIPLE GROUPING COLUMNS
    # =========================================================

    def group_by_multiple(
        self,
        group_by,
        value_column,
        aggregation="sum",
        filters=None,
        sort="desc",
        limit=30
    ):
        """
        Groups using multiple columns.

        Example:
        Company + Material
        Company + Number
        Material + Number
        """

        if not isinstance(group_by, list):
            return {
                "error": "'group_by' must be a list of columns."
            }

        for column in group_by:
            if column not in self.enriched_dataset.columns:
                return {
                    "error": f"Column '{column}' does not exist."
                }

        if value_column not in self.enriched_dataset.columns:
            return {
                "error": f"Column '{value_column}' does not exist."
            }

        allowed_aggregations = {
            "sum",
            "mean",
            "count",
            "min",
            "max",
            "median"
        }

        if aggregation not in allowed_aggregations:
            return {
                "error": "Unsupported aggregation."
            }

        dataset = self._apply_filters(
            self.enriched_dataset,
            filters
        )

        dataset = dataset.copy()

        if aggregation != "count":
            dataset[value_column] = pd.to_numeric(
                dataset[value_column],
                errors="coerce"
            )

        result = (
            dataset
            .groupby(group_by, dropna=False)[value_column]
            .agg(aggregation)
            .reset_index()
        )

        result_column = f"{aggregation}_{value_column}"

        result = result.rename(
            columns={
                value_column: result_column
            }
        )

        result = (
            result
            .sort_values(
                result_column,
                ascending=sort == "asc"
            )
            .head(limit)
        )

        return result.to_dict(orient="records")

    # =========================================================
    # COUNTING
    # =========================================================

    def value_counts(self, column, filters=None, limit=20):

        if not self._validate_column(column):
            return {
                "error": f"Column '{column}' does not exist."
            }

        dataset = self._apply_filters(
            self.enriched_dataset,
            filters
        )

        result = (
            dataset[column]
            .value_counts(dropna=False)
            .head(limit)
        )

        return [
            {
                column: str(index),
                "count": int(count)
            }
            for index, count in result.items()
        ]

    # =========================================================
    # CARD INFORMATION
    # =========================================================

    def get_card_info(self, card_number):

        if "Number" not in self.cards.columns:
            return {
                "error": "Card Number column does not exist."
            }

        result = self.cards[
            self.cards["Number"].astype(str)
            == str(card_number)
        ]

        if result.empty:
            return {
                "found": False,
                "card_number": str(card_number)
            }

        columns = [
            column
            for column in [
                "Number",
                "Company",
                "Name"
            ]
            if column in result.columns
        ]

        return result[columns].to_dict(
            orient="records"
        )

    # =========================================================
    # TRANSACTION SEARCH
    # =========================================================

    def search_transactions(
        self,
        filters=None,
        columns=None,
        sort_by=None,
        sort="desc",
        limit=50
    ):
        """
        Searches transactions without sending the entire dataset
        to the LLM.
        """

        dataset = self._apply_filters(
            self.enriched_dataset,
            filters
        )

        if sort_by:

            if sort_by not in dataset.columns:
                return {
                    "error": f"Column '{sort_by}' does not exist."
                }

            dataset = dataset.sort_values(
                sort_by,
                ascending=sort == "asc"
            )

        if columns:

            valid_columns = [
                column
                for column in columns
                if column in dataset.columns
            ]

            dataset = dataset[valid_columns]

        dataset = dataset.head(limit)

        return dataset.to_dict(
            orient="records"
        )

    def get_transactions_by_card(
        self,
        card_number,
        limit=100
    ):

        return self.search_transactions(
            filters=[
                {
                    "column": "Number",
                    "operator": "eq",
                    "value": str(card_number)
                }
            ],
            limit=limit
        )

    # =========================================================
    # DISTINCT COUNTS
    # =========================================================

    def count_unique(self, column, filters=None):

        if not self._validate_column(column):
            return {
                "error": f"Column '{column}' does not exist."
            }

        dataset = self._apply_filters(
            self.enriched_dataset,
            filters
        )

        return {
            "column": column,
            "unique_count": int(
                dataset[column].nunique(dropna=True)
            )
        }

    # =========================================================
    # FIND EXTREME VALUE
    # =========================================================

    def find_extreme(
        self,
        group_by,
        value_column,
        aggregation="sum",
        extreme="max",
        filters=None
    ):
        """
        Directly finds the highest or lowest aggregated value.
        """

        result = self.group_and_aggregate(
            group_by=group_by,
            value_column=value_column,
            aggregation=aggregation,
            filters=filters,
            sort="desc" if extreme == "max" else "asc",
            limit=1
        )

        return result


agent_instance = AgentBaseFunctions(
    df,
    client_cards
)