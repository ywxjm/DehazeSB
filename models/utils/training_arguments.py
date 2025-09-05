from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):

    cfg: str = None
    
    # seed: int = field(
    #     default=42,
    #     metadata={"help": "Random seed for initialization"}
    # )

    verbose: bool = field(
        default=False,
    )



