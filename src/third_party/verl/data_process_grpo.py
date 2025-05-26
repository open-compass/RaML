import os
import copy
import random

import datasets


if __name__ == "__main__":
    random.seed(42)

    dataset = datasets.load_dataset("jnanliu/orz-math-filtered", split="train")
    dataset = dataset.filter(lambda example: str.isdigit(example["answer"]))
    datset = dataset.map(lambda example: {"answer": [example["answer"]]})

    instruction = """A conversation between a User and an Assistant. The User poses a question, and the Assistant provides a solution. The Assistant's response follows these structured steps:
    
1. **Reasoning Process**: The Assistant reflects on the problem using a reasoning process enclosed within `<think>` and `</think>` tags.
2. **Conclusion**: The Assistant reaches a conclusion, which is enclosed within `<conclusion>` and `</conclusion>` tags. The final answer is highlighted within `\\boxed{...final answer...}`.
3. **Answer Format**: The complete response should be formatted as:

<think>
...reasoning process...
</think>
<conclusion>
...conclusion...
The answer is \\boxed{...final answer...}
</conclusion>""".strip()

    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop("problem")
            answer = example.pop("answer")
            data = {
                "data_source": "orz-math-filtered",
                "prompt": [
                    {
                        "role": "system",
                        "content": instruction
                    },
                    {
                        "role": "user",
                        "content": problem
                    }
                ],
                "answer": answer,
                "ability": "math",
                "data_source": "orz-math",
                "data_type": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    "split": split,
                    "index": idx
                }
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("train"), with_indices=True)
    dataset.to_parquet(os.path.join("data/rl", f"train.parquet"))

    val_dataset = []
    math_dataset = datasets.load_dataset("jnanliu/orz-math-filtered-val", split="test")
    for example in math_dataset:
        val_dataset.append(
            {
                "question": example["problem"],
                "answer": [example["answer"]],
                "data_type": "math",
                "data_source": "orz_math"
            }
        )


    instruction = """A conversation between a User and an Assistant. The User poses a question, and the Assistant provides a solution. The Assistant's response follows these structured steps:
    
1. **Reasoning Process**: The Assistant reflects on the problem using a reasoning process enclosed within `<think>` and `</think>` tags.
2. **Conclusion**: The Assistant reaches a conclusion, which is enclosed within `<conclusion>` and `</conclusion>` tags. The final answer is highlighted within `\\boxed{...final answer...}`.
3. **Answer Format**: The complete response should be formatted as:

<think>
...reasoning process...
</think>
<conclusion>
...conclusion...
The answer is \\boxed{...final answer...}
</conclusion>""".strip()

    dataset = datasets.Dataset.from_list(val_dataset)
    dataset = dataset.map(function=make_map_fn("val"), with_indices=True)
    dataset.to_parquet(os.path.join("data/rl", f"val.parquet"))
