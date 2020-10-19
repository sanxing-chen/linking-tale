import json
import shutil
import sys
import argparse

from allennlp.commands import main

parser = argparse.ArgumentParser()
parser.add_argument("path", help="The folder to store the output of the experiment")
parser.add_argument('--gated', action='store_true', help="Enable the proposed dynamic gates")
parser.add_argument('--mode', choices=['train', 'eval', 'recover'], default='train',
                    help="Train, evaluate or recover a model")
parser.add_argument('--ablation', choices=['wo_copy', 'ent_rem', 'wo_reuse_emb'],
                    help="Train, evaluate or recover a model")
args = parser.parse_args()

experiment_path = args.path

overrides = {"model": {"log_path": experiment_path}}
if args.gated:
    overrides["model"]["enable_gating"] = True
if args.ablation:
    overrides["model"]["ablation_mode"] = args.ablation
overrides = json.dumps(overrides)

if args.mode != 'eval':
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        "train_configs/defaults.jsonnet",
        "-s", experiment_path,
        "--include-package", "dataset_readers.spider",
        "--include-package", "models.semantic_parsing.spider_parser",
        "-o", overrides,
    ]

    # Might need to set seeds manually 
    if args.mode == 'recover':
        sys.argv.append("--recover")
else:
    sys.argv = [
        "allennlp",
        "predict", experiment_path,
        "spider/dev.json",
        "--predictor", "spider",
        "--use-dataset-reader",
        "--cuda-device=0",
        "--silent",
        "--output-file", experiment_path + "/prediction-t.sql",
        "--include-package", "models.semantic_parsing.spider_parser",
        "--include-package", "dataset_readers.spider",
        "--include-package", "predictors.spider_predictor",
        "--weights-file", experiment_path + "/best.th",
        "--batch-size", "15",
        "-o", "{\"dataset_reader\":{\"keep_if_unparsable\":true}, \"validation_iterator\":{\"batch_size\": 15}}"
    ]

main()
