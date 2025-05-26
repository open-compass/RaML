import os

from mmengine.config import read_base
with read_base():
    from opencompass.configs.datasets.livecodebench.livecodebench_gen_6966bc import LCBCodeGeneration_dataset

from opencompass.models import TurboMindModelwithChatTemplate, VLLMwithChatTemplate
from opencompass.datasets.aime2024 import Aime2024Dataset
from opencompass.datasets.aime2025 import Aime2025Dataset
from opencompass.datasets.math import MATH500Dataset
from opencompass.datasets.gpqa import GPQADataset, GPQAEvaluator
from opencompass.datasets.livemathbench import LiveMathBenchDataset

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator.math_evaluator import MATHEvaluator


SYSTEM_PROMPT = """A conversation between a User and an Assistant. The User poses a question, and the Assistant provides a solution. The Assistant's response follows these structured steps:

1. **Reasoning Process**: The Assistant comprehensively thinks about the problem through a reasoning process.
2. **Conclusion**: The Assistant reaches a conclusion, which is enclosed within `<conclusion>` and `</conclusion>` tags. The final answer is highlighted within `\\boxed{...final answer...}`.
3. **Response Format**: The complete response should be formatted as:

...reasoning process...
<conclusion>
...conclusion...
The answer is \\boxed{...final answer...}
</conclusion>
""".strip()

models = [
    dict(
        type=VLLMwithChatTemplate,
        path=path,
        model_kwargs=dict(
            trust_remote_code=True,
            tensor_parallel_size=num_gpu,
            gpu_memory_utilization=0.8
        ),
        generation_kwargs=dict(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=-1
        ),
        meta_template=dict(
            reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM'),],
            round=[
                dict(role='HUMAN', api_role='HUMAN'),
                dict(role='BOT', api_role='BOT', generate=True),
            ]
        ),
        abbr=abbr,     
        max_out_len=32768,
        batch_size=4096,
        run_cfg=dict(num_gpus=num_gpu)
    )
    for abbr, path, num_gpu in [
        ("Qwen/Qwen2.5-7B-Instruct", 4),
    ]
]

aime24_dataset = dict(
    type=Aime2024Dataset,
    k=8,
    n=32,
    abbr='aime24',
    reader_cfg=dict(
        input_columns=['question'], 
        output_column='answer'
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(role='SYSTEM', prompt=SYSTEM_PROMPT),
                ],
                round=[
                    dict(role='HUMAN', prompt='{question}'),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=MATHEvaluator
        )
    )
)
aime25_dataset = dict(
    type=Aime2025Dataset,
    k=8,
    n=32,
    abbr='aime25',
    reader_cfg=dict(
        input_columns=['question'], 
        output_column='answer'
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(role='SYSTEM', prompt=SYSTEM_PROMPT),
                ],
                round=[
                    dict(role='HUMAN', prompt='{question}'),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=MATHEvaluator
        )
    )
)
math500_dataset = dict(
    type=MATH500Dataset,
    k=8,
    n=32,
    abbr='math500',
    reader_cfg=dict(
        input_columns=['question'], 
        output_column='answer'
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(role='SYSTEM', prompt=SYSTEM_PROMPT),
                ],
                round=[
                    dict(role='HUMAN', prompt='{question}'),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=16384
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=MATHEvaluator
        )
    )
)
gpqa_dataset = dict(
    type=GPQADataset,
    k=8,
    n=32,
    abbr='gpqa-diamond',
    reader_cfg=dict(
        input_columns=['question', 'a', 'b', 'c', 'd'], 
        output_column='answer'
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(role='SYSTEM', prompt=SYSTEM_PROMPT),
                ],
                round=[
                    dict(role='HUMAN', prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer, 
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=GPQAEvaluator
        )
    )
)
livemathbench_hard_dataset = dict(
    type=LiveMathBenchDataset,
    path='opencompass/LiveMathBench',
    k=8,
    n=32,
    abbr='livemathbench_hard',
    dataset_splits=['hard'],
    dataset_languages=['cn', 'en'],
    cot=True,
    version='202412',
    reader_cfg=dict(
        input_columns=['prompt'], 
        output_column='answer'
    ),
    infer_cfg=dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(role='SYSTEM', prompt=SYSTEM_PROMPT),
                ],
                round=[
                    dict(role='HUMAN', prompt='{prompt}'),
                ]
            )
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(
            type=GenInferencer,
            max_out_len=32768
        ),
    ),
    eval_cfg=dict(
        evaluator=dict(
            type=MATHEvaluator
        )
    )
)
livecodebench_dataset = LCBCodeGeneration_dataset
livecodebench_dataset.update(dict(k=8, n=32, abbr='livecodebench', release_version="v4_v5"))
livecodebench_dataset["infer_cfg"]["inferencer"].update(dict(max_out_len=8192))
livecodebench_dataset["eval_cfg"]["evaluator"].update(dict(release_version="v4_v5", extractor_version="v2"))

datasets = [aime24_dataset, aime25_dataset, math500_dataset]