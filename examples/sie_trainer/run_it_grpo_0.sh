set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

# 指定上下文模式，请修改这里，可以是 [0-6] 的数字
context_mode=0
experiment_name="sie_qwen2.5_7b_it_grpo_${context_mode}"

HOME="your/home/path"

gsm8k_test_path=$HOME/data/gsm8k_test.parquet
math500_test_path=$HOME/data/math500_test.parquet

kk_test_easy_path=$HOME/data/kk_test_easy.parquet
kk_test_hard_path=$HOME/data/kk_test_hard.parquet

webqsp_train_path=$HOME/data/webqsp_${context_mode}_train.parquet
webqsp_dev_path=$HOME/data/webqsp_${context_mode}_validation.parquet
webqsp_test_path=$HOME/data/webqsp_${context_mode}_test.parquet

cwq_train_path=$HOME/data/cwq_${context_mode}_train.parquet
cwq_dev_path=$HOME/data/cwq_${context_mode}_validation.parquet
cwq_test_path=$HOME/data/cwq_${context_mode}_test.parquet

grailqa_test_path=$HOME/data/grailqa_${context_mode}_test.parquet

train_files="['$webqsp_train_path', '$webqsp_dev_path', '$cwq_train_path']"
test_files="['$webqsp_test_path', '$cwq_test_path', '$grailqa_test_path', '$gsm8k_test_path', '$math500_test_path', '$kk_test_easy_path', '$kk_test_hard_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.project_name='sie_trainer' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=600 \
    2>&1 | tee "${experiment_name}.log"
