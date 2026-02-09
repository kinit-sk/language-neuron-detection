from runexp import TrackedExperiment

class Experiment(TrackedExperiment):
    def __call__(self):
        trainer = self.config_built.trainer
        trainer.train()

def take(dataset, num_take):
    return dataset.take(num_take)

def shard_take(dataset, num_shards, index, num_take):
    return dataset.shard(num_shards=num_shards, index=index).take(num_take)

def load_tokenizer(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
