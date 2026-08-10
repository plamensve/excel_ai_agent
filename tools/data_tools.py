import pandas as pd

df = pd.read_excel("../data/data.xlsx")


class AgentBaseFunctions:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_dataset_shape(self):
        return {
            "rows": self.dataset.shape[0],
            "columns": self.dataset.shape[1]
        }

    def get_columns(self):
        return self.dataset.columns.tolist()

    def get_missing_values(self):
        return self.dataset.isna().sum().to_dict()

    def calculate_mean(self, column):
        if column not in self.dataset.columns:
            return {
                "Error": f"Column '{column}' does not exist."
            }
        values = pd.to_numeric(self.dataset[column], errors='coerce')
        return float(values.mean())

    def get_value_count(self, column):
        return self.dataset[column].value_counts().to_dict()

test_inst = AgentBaseFunctions(df)
print(test_inst.get_dataset_shape())
