import trax
import numpy as np
from trax import layers as tl

from trax.supervised import training
import os

model = tl.Serial(
    tl.Embedding(vocab_size=8192, d_feature=256),
    tl.Mean(axis=1),  # Average on axis 1 (length of sentence).
    tl.Dense(2),  # Classify 2 classes.
    tl.LogSoftmax()  # Produce log-probabilities.
)

train_stream = trax.data.TFDS('imdb_reviews',
                              keys=('text', 'label'),
                              train=True)()
eval_stream = trax.data.TFDS('imdb_reviews',
                             keys=('text', 'label'),
                             train=False)()

data_pipeline = trax.data.Serial(
    trax.data.Tokenize(vocab_file='en_8k.subword', keys=[0]),
    trax.data.Shuffle(),
    trax.data.FilterByLength(max_length=2048, length_keys=[0]),
    trax.data.BucketByLength(boundaries=[32, 128, 512, 2048],
                             batch_sizes=[256, 64, 16, 4, 1],
                             length_keys=[0]), trax.data.AddLossWeights())
train_batches_stream = data_pipeline(train_stream)
eval_batches_stream = data_pipeline(eval_stream)
example_batch = next(train_batches_stream)

# Training task.
train_task = training.TrainTask(
    labeled_data=train_batches_stream,
    loss_layer=tl.WeightedCategoryCrossEntropy(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=500,
)

# Evaluaton task.
eval_task = training.EvalTask(
    labeled_data=eval_batches_stream,
    metrics=[tl.WeightedCategoryCrossEntropy(),
             tl.WeightedCategoryAccuracy()],
    n_eval_batches=20  # For less variance in eval numbers.
)

# Training loop saves checkpoints to output_dir.
output_dir = os.path.expanduser('~/output_dir/')
training_loop = training.Loop(model,
                              train_task,
                              eval_tasks=[eval_task],
                              output_dir=output_dir)

# Run 2000 steps (batches).
training_loop.run(200)

example_input = next(eval_batches_stream)[0][0]
example_input_str = trax.data.detokenize(example_input,
                                         vocab_file='en_8k.subword')
print(f'example input_str: {example_input_str}')
sentiment_log_probs = model(example_input[None, :])  # Add batch dimension.
print(f'Model returned sentiment probabilities: {np.exp(sentiment_log_probs)}')
