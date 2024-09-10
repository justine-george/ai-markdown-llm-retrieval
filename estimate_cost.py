import tiktoken
import os
import argparse

embedding_cost_per_million_tokens = {
    "text-embedding-3-small": 0.02,
}

prompt_cost_per_million_tokens = {
    "gpt-3.5-turbo": {
        "input": 3,
        "output": 6,
        "training": 8,
    },
}


def get_num_tokens(data_folder, model):
    encoding = tiktoken.encoding_for_model(model)

    total_tokens = 0
    files_found = False
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".md"):
                files_found = True
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    total_tokens += len(encoding.encode(content))
                except IOError as e:
                    print(f"Error reading file {file_path}: {e}")

    if not files_found:
        return (0, -1)

    return (total_tokens, 0)


def estimate_embedding_cost(data_folder, model):
    num_tokens, error_code = get_num_tokens(data_folder, model)
    print(f"Number of tokens: {num_tokens}")
    if error_code == -1:
        return (0, -1)

    cost = num_tokens * embedding_cost_per_million_tokens[model] / 1_000_000
    return (cost, 0)


def get_num_tokens_for_prompt(prompt_text, model):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt_text))
    return (num_tokens, 0)


def estimate_prompt_cost(prompt_text, model, prompt_type):
    if prompt_type not in prompt_cost_per_million_tokens[model]:
        return (0, -1)

    num_tokens, error_code = get_num_tokens_for_prompt(prompt_text, model)
    if error_code == -1:
        return (0, -1)
    cost = num_tokens * \
        prompt_cost_per_million_tokens[model][prompt_type] / 1_000_000
    return (cost, 0)


if __name__ == "__main__":
    # https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to the folder containing data files")
    args = parser.parse_args()

    model = "text-embedding-3-small"
    data_folder = args.data_folder

    cost, error_code = estimate_embedding_cost(data_folder, model)
    if error_code == -1:
        print("No files found in the data folder")
    else:
        print(f"Estimated embedding cost: ${cost:.3f}")
