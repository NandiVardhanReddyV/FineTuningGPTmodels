from datasets import load_dataset

def preprocess_data(output_dir):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    dataset["train"].to_csv(f"{output_dir}/train.csv")
    dataset["validation"].to_csv(f"{output_dir}/validation.csv")
    dataset["test"].to_csv(f"{output_dir}/test.csv")

if __name__ == "__main__":
    preprocess_data("data")