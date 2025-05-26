import os
import copy
import random

import datasets


if __name__ == "__main__":
    random.seed(42)

    ngs = [1, 2, 4, 8, 16, 32, 64]
    # data_source = "jnanliu/orz-math-filtered-qwen-72b-rollout"
    data_source = "jnanliu/orz-math-filtered-distill-14b-rollout"
    local_dir = "data/sft/"
    
    os.makedirs(local_dir, exist_ok=True)

    dataset = datasets.load_dataset(data_source)["train"]

    for ng in ngs:
        train_dataset = []
        for example in dataset:
            all_generations = [(g, l) for g, l in zip(example["generations"], example["labels"])]
            all_generations = sorted(all_generations, key=lambda x: -x[1])
            current_examples = []
            for g, l in all_generations[:ng]:
                if l == 0:
                    continue
                ex = copy.deepcopy(example)
                ex.pop("generations")
                ex.pop("labels")
                ex.update({"generation": g})
                train_dataset.append(ex)
                current_examples.append(ex)
            repeat_num = sum([l == 1 for g, l in all_generations[:max(ngs)]])
            for i in range(len(current_examples), repeat_num):
                train_dataset.append(random.choice(current_examples))
        
        random.shuffle(train_dataset)
        train_dataset = datasets.Dataset.from_list(train_dataset)
        print("len of dataset: ", len(train_dataset))

        if "qwen-72b" in data_source:
            instruction = "Please solve the following mathematical problem step by step and put your final answer in \\boxed{}. Here is the problem: \n\n"
        if "distill" in data_source:
            instruction = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

        def make_map_fn(split):

            def process_fn(example, idx):
                problem = example.pop("problem")

                answer = example.pop("answer")
                generation = example.pop("generation")
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "system",
                            "content": instruction
                        },
                        {
                            "role": "user",
                            "content": problem
                        }
                    ] if "distill" in data_source else [
                        {
                            "role": "user",
                            "content": instruction + problem 
                        }
                    ],
                    "solution": generation,
                    "answer": answer,
                    "ability": "math",
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

        train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
        train_dataset.to_parquet(os.path.join(local_dir, f"ng{ng}/train.parquet"))
