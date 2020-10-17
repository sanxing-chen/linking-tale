local dataset_path = "spider/";
local bert_model = "bert-base-uncased";
local question_token_indexers = {
  "tokens": {
    "type": "bert-pretrained",
    "pretrained_model": bert_model,
  }
};

{
  "random_seed": 11731,
  "numpy_seed": 11731,
  "pytorch_seed": 11731,
  "dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "keep_if_unparsable": false,
    "loading_limit": -1,
    "question_token_indexers": question_token_indexers
  },
  "validation_dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "keep_if_unparsable": true,
    "loading_limit": -1,
    "question_token_indexers": question_token_indexers
  },
  "train_data_path": dataset_path + "train_spider.json",
  "validation_data_path": dataset_path + "dev.json",
  "model": {
    "type": "spider",
    "log_path": "",
    "enable_gating": false,
    "dataset_path": dataset_path,
    "parse_sql_on_decoding": true,
    "gnn": true,
    "gnn_timesteps": 3,
    "pruning_gnn_timesteps": 3,
    "decoder_self_attend": true,
    "decoder_use_graph_entities": true,
    "use_neighbor_similarity_for_linking": true,
    "question_embedder": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": bert_model,
        "requires_grad": true,
        "top_layer_only": true
      },
      "embedder_to_indexer_map": {
        "bert": {
            "input_ids": "tokens",
            "offsets": "tokens-offsets",
            "token_type_ids": "tokens-type-ids"
        }
      },
      "allow_unmatched_keys": true
    },
    "action_embedding_dim": 200,
    "encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": 400,
      "bidirectional": true,
      "num_layers": 1
    },
    "entity_encoder": {
      "type": "boe",
      "embedding_dim": 200,
      "averaged": true
    },
    "decoder_beam_search": {
      "beam_size": 10
    },
    "training_beam_size": 1,
    "max_decoding_steps": 100,
    "input_attention": {"type": "dot_product"},
    "past_attention": {"type": "additive", "vector_dim": "201", "matrix_dim": "201", "normalize": true},
    "graph_attention": {"type": "additive", "vector_dim": "200", "matrix_dim": "200", "normalize": false},
    "dropout": 0.1
  },
  "iterator": {
    "type": "basic",
    "track_epoch": true,
    "batch_size" : 8
  },
  "validation_iterator": {
    "type": "basic",
    "track_epoch": true,
    "batch_size" : 1
  },
  "trainer": {
    "num_epochs": 100,
    "cuda_device": 0,
    "patience": 50,
    "validation_metric": "+sql_match",
    "optimizer": {
      "type": "bert_adam",
      "lr": 1e-3,
      "weight_decay": 5e-4,
      "max_grad_norm": 1.0,
      "t_total": 90000,
      "warm": 0.02,
      "parameter_groups": [
        [["bert_model.*(?<!Norm|norm).weight"], {
          "lr": 1e-5,
          "t_total": 5000,
          "warm": 0.3,
          "weight_decay": 0.01,
        }],
        [["bert_model.*bias", "bert_model.*LayerNorm.bias", "bert_model.*LayerNorm.weight"], {
          "lr": 1e-5,
          "t_total": 5000,
          "warm": 0.3,
          "weight_decay": 0.0
        }]
      ]
    },
    "num_serialized_models_to_keep": 2
  }
}
