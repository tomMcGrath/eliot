import config


# Base config
model_cfg = config.ModelConfig(
    num_layers=1,
    d_model=1, 
    n_heads=1,
    d_head=1,
    d_mlp=1,
    vocab_size=1,
    tokenizer='gpt2', 
    max_len=1,
    )

dataset_cfg = config.DatasetConfig(
    train_dataset='wikitext',
    seq_len=1,
    batch_size=1,
    tokenizer='gpt2',
    )

training_cfg = config.TrainingConfig(
    warmup_steps=1,
    post_warmup_steps=1,
    lr_max=1,
    lr_min=1,
    weight_decay=1.,
    adam_beta1=1.,
    adam_beta2=1.
    )


def check_err_prefix_in_list(err_prefix, err_list):
    """Helper function to check if some prefix of an error 
    (e.g. "Tokenizer mismatch") is in a list of errors."""
    for err in err_list:
        if err.startswith(err_prefix):
            return True
        
    return False


class TestPreflight:

    def test_catches_tokenizer_mismatch(self):
        dataset_cfg_with_wrong_tokenizer = config.DatasetConfig(
            train_dataset='wikitext',
            seq_len=1,
            batch_size=1,
            tokenizer='gpt4',
            )

        cfg = config.Config(
            model_config=model_cfg,
            training_config=training_cfg,
            dataset_config=dataset_cfg_with_wrong_tokenizer,
            )
        
        preflight_results = cfg.passes_prelaunch_checks()

        assert not preflight_results.passes
        assert check_err_prefix_in_list(
            'Tokenizer mismatch', preflight_results.errs)
        
    def test_catches_seq_len_mismatch(self):
        dataset_cfg_with_wrong_seq_len = config.DatasetConfig(
            train_dataset='wikitext',
            seq_len=2,
            batch_size=1,
            tokenizer='gpt4',
            )
        
        cfg = config.Config(
            model_config=model_cfg,
            training_config=training_cfg,
            dataset_config=dataset_cfg_with_wrong_seq_len,
            )
        
        preflight_results = cfg.passes_prelaunch_checks()

        assert not preflight_results.passes
        assert check_err_prefix_in_list(
            'Sequence length mismatch', preflight_results.errs)
        
    def test_passes_when_correct(self):
        cfg = config.Config(
            model_config=model_cfg,
            training_config=training_cfg,
            dataset_config=dataset_cfg,
            )
        
        preflight_results = cfg.passes_prelaunch_checks()
        assert preflight_results.passes


class TestTrainingConfig:

    def test_step_count_helper(self):
        assert training_cfg.total_num_steps == 2
