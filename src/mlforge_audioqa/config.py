from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    sample_rate: int = 16000
    window_seconds: float = 1.5
    hop_seconds: float = 0.5
    min_event_duration: float = 0.2
    event_merge_gap: float = 0.4
    default_threshold: float = 0.15
    top_k_per_window: int = 5
    # Open-vocabulary environmental sound bank used by CLAP/AST matching.
    labels: list[str] = field(
        default_factory=lambda: [
            "bird chirping",
            "train passing",
            "drilling",
            "traffic noise",
            "car passing",
            "car horn",
            "dog bark",
            "cat meowing",
            "rain",
            "thunder",
            "wind",
            "water flowing",
            "footsteps",
            "door slam",
            "glass breaking",
            "construction",
            "construction noise",
            "hammering",
            "siren",
            "ambulance siren",
            "police siren",
            "street music",
            "crowd talking",
            "speech",
            "whisper",
            "baby crying",
            "laughing",
            "clapping",
            "drumming",
            "keyboard typing",
            "phone ringing",
            "alarm",
            "machine hum",
            "factory noise",
            "helicopter",
            "airplane",
            "motorcycle",
            "bus passing",
            "subway",
            "bell",
        ]
    )
    per_label_threshold: dict[str, float] = field(
        default_factory=lambda: {
            # Lower thresholds for underperforming transient-impact classes.
            "drilling": 0.08,
            "hammering": 0.08,
            # Slightly higher thresholds for frequent false positives.
            "car passing": 0.21,
            "car horn": 0.22,
            "traffic noise": 0.18,
        }
    )


DEFAULT_CONFIG = PipelineConfig()
