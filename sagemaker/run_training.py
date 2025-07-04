# Launch SDXL LoRA training on SageMaker
# TODO: Fill in your IAM role and S3 paths

from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='train_lora_sdxl.py',
    source_dir='..',
    instance_type='ml.g5.2xlarge',
    instance_count=1,
    role='YOUR_SAGEMAKER_ROLE',
    transformers_version='4.36',
    pytorch_version='2.1',
    py_version='py310',
    base_job_name='style-lora-sdxl'
)

huggingface_estimator.fit({'train': 's3://your-bucket/path-to-style-data'})
