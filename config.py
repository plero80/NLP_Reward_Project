@dataclass(frozen=True)
class ClassifierTrainingConfig:
    output_dir: str
    epochs: float = 3.0
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    lora_settings: LoRASettings | None = None