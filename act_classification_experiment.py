import traceback
from act_classification_fine_tune import fine_tune_model



def get_base_params():
    return {
        "model_name": "distilbert-base-uncased",
        "label_column": "general_da",

        "max_weight": 2.5,
        "min_weight": 0.6,

        "lora_dropout": 0.15,
        "lora_r": 64,
        "lora_alpha": 128,
        "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"],

        "no_epochs": 3,
        "batch_size": 16,
        "learning_rate": 0.00006,
        "warmup_steps": 0.15,
        "weight_decay": 0.05,
        
        "loss_type": "focal_weighted",
        "focal_gamma": 3.5,
        "loss_reduction": "mean",

        "experiment_name": "",
        "experiment_description": "",
    }

def get_experiments():
    # experiments = [
    #     {
    #         "name": "exp01_baseline",
    #         "description": "Baseline configuration",
    #         "params": {}
    #     },
    #     {
    #         "name": "exp02_high_lr",
    #         "description": "Higher learning rate",
    #         "params": {
    #             "learning_rate": 0.0002
    #         }
    #     },
    #     {
    #         "name": "exp03_low_lr", 
    #         "description": "Lower learning rate",
    #         "params": {
    #             "learning_rate": 0.00003
    #         }
    #     },
    #     {
    #         "name": "exp04_small_lora",
    #         "description": "Smaller LoRA rank for efficiency",
    #         "params": {
    #             "lora_r": 32,
    #             "lora_alpha": 64
    #         }
    #     },
    #     {
    #         "name": "exp05_large_lora",
    #         "description": "Larger LoRA rank for capacity",
    #         "params": {
    #             "lora_r": 128,
    #             "lora_alpha": 256
    #         }
    #     },
    #     {
    #         "name": "exp06_ce_weighted",
    #         "description": "Cross-entropy with class weights",
    #         "params": {
    #             "loss_type": "ce_weighted"
    #         }
    #     },
    #     {
    #         "name": "exp07_focal_only",
    #         "description": "Focal loss without class weights",
    #         "params": {
    #             "loss_type": "focal"
    #         }
    #     },
    #     {
    #         "name": "exp08_high_focal_gamma",
    #         "description": "Higher focal gamma for hard examples",
    #         "params": {
    #             "focal_gamma": 3.5
    #         }
    #     },
    #     {
    #         "name": "exp09_large_batch",
    #         "description": "Larger batch size",
    #         "params": {
    #             "batch_size": 64,
    #             "learning_rate": 0.00012  # Scale LR with batch size
    #         }
    #     },
    #     {
    #         "name": "exp10_small_batch",
    #         "description": "Smaller batch size for better gradients",
    #         "params": {
    #             "batch_size": 16,
    #             "learning_rate": 0.00006  # Scale down LR with smaller batch
    #         }
    #     },
    #     {
    #         "name": "exp11_heavy_regularization",
    #         "description": "Strong regularization",
    #         "params": {
    #             "weight_decay": 0.1,
    #             "warmup_steps": 0.25,
    #             "lora_dropout": 0.2,
    #         }
    #     },
    #     {
    #         "name": "exp12_optimized_combo",
    #         "description": "Best combination from previous experiments",
    #         "params": {
    #             "learning_rate": 0.0001,
    #             "lora_r": 64,
    #             "lora_alpha": 128,
    #             "focal_gamma": 2.5,
    #             "weight_decay": 0.08,
    #             "warmup_steps": 0.20,
    #         }
    #     },
    #     {
    #         "name": "exp13_high_weight_ce",
    #         "description": "Cross-entropy with stronger class weights",
    #         "params": {
    #             "loss_type": "ce_weighted",
    #             "max_weight": 15,
    #             "min_weight": 0.2
    #         }
    #     },
    #     {
    #         "name": "exp14_ce_weighted_extreme",
    #         "description": "Cross-entropy with extreme weights",
    #         "params": {
    #             "loss_type": "ce_weighted",
    #             "max_weight": 20,
    #             "min_weight": 0.1
    #         }
    #     },  
    #     {
    #         "name": "exp15_extreme_focal",
    #         "description": "Focal loss with higher gamma",
    #         "params": {
    #             "loss_type": "focal",
    #             "focal_gamma": 4.0
    #         }
    #     },
    #     {
    #         "name": "exp16_strong_focal_weighted_strong",
    #         "description": "Strong focal loss with stronger weights",
    #         "params": {
    #             "loss_type": "focal_weighted",
    #             "focal_gamma": 3.0,
    #             "max_weight": 12.5,
    #             "min_weight": 0.25
    #         }
    #     },
    #     {
    #         "name": "exp17_strong_focal_weighted_weak",
    #         "description": "Strong focal loss with weak class weights",
    #         "params": {
    #             "loss_type": "focal_weighted",
    #             "focal_gamma": 3.0,
    #             "max_weight": 5,
    #             "min_weight": 0.5
    #         }
    #     },
    #     {
    #         "name": "exp17_base_focal_weighted_strong",
    #         "description": "Base focal loss with strong class weights",
    #         "params": {
    #             "loss_type": "focal_weighted",
    #             "focal_gamma": 2.0,
    #             "max_weight": 15,
    #             "min_weight": 0.2
    #         }
    #     },
    #     {
    #         "name": "exp19_low_focal_weighted_strong",
    #         "description": "Low focal loss with strong class weights",
    #         "params": {
    #             "loss_type": "focal_weighted",
    #             "focal_gamma": 1.5,
    #             "max_weight": 15,
    #             "min_weight": 0.2
    #         }
    #     },
    #     {
    #         "name": "exp20_extreme_focal_weighted_strong",
    #         "description": "Low focal loss with strong class weights",
    #         "params": {
    #             "loss_type": "focal_weighted",
    #             "focal_gamma": 4,
    #             "max_weight": 15,
    #             "min_weight": 0.2
    #         }
    #     },
    #     {
    #         "name": "exp21_extreme_focal_weighted_base",
    #         "description": "Low focal loss with strong class weights",
    #         "params": {
    #             "loss_type": "focal_weighted",
    #             "focal_gamma": 4,
    #         }
    #     },
    #     {
    #         "name": "exp22_extreme_focal_weighted_weak",
    #         "description": "Low focal loss with strong class weights",
    #         "params": {
    #             "loss_type": "focal_weighted",
    #             "focal_gamma": 4,
    #             "max_weight": 5,
    #             "min_weight": 0.5
    #         }
    #     },
    #     {
    #         "name": "exp23_extreme_focal_batch_16_lr_0.00006",
    #         "description": "Extreme focal loss with small batch and low learning rate",
    #         "params": {
    #             "loss_type": "focal",
    #             "focal_gamma": 4,
    #             "batch_size": 16,
    #             "learning_rate": 0.00006
    #         }
    #     },
    #     {
    #         "name": "exp24_strong_focal_weighted_weak_batch_16_lr_0.00006",
    #         "description": "Strong focal loss with weak class weights, small batch and low learning rate",
    #         "params": {
    #             "loss_type": "focal_weighted",
    #             "focal_gamma": 3.5,
    #             "batch_size": 16,
    #             "learning_rate": 0.00006,
    #             "max_weight": 2.5,
    #             "min_weight": 0.6
    #         }
    #     },
    # ]
    experiments = [
        {
            "experiment_name": "bert_exp01_baseline",
            "experiment_description": "Baseline configuration",
            "model_name": "bert-base-uncased",
            "target_modules": ["query", "key", "value", "dense"]
        },
        {
            "experiment_name": "roberta_exp01_baseline",
            "experiment_description": "Baseline configuration",
            "model_name": "roberta-base",
            "target_modules": ["query", "key", "value", "dense"]
        },
        {
            "experiment_name": "5_exp01_baseline",
            "experiment_description": "Baseline configuration 5 classes",
            "label_column": "basic_da",
        },
        {
            "experiment_name": "5_bert_exp01_baseline",
            "experiment_description": "Baseline configuration 5 classes",
            "label_column": "basic_da",
            "model_name": "bert-base-uncased",
            "target_modules": ["query", "key", "value", "dense"],
        },
        {
            "experiment_name": "5_roberta_exp01_baseline",
            "experiment_description": "Baseline configuration 5 classes",
            "label_column": "basic_da",
            "model_name": "roberta-base",
            "target_modules": ["query", "key", "value", "dense"],
        },
        {
            "experiment_name": "5_bert_exp02_lower_penalty",
            "experiment_description": "Lower penalty for less imbalanced dataset",
            "label_column": "basic_da",
            "model_name": "bert-base-uncased",
            "target_modules": ["query", "key", "value", "dense"],
            "max_weight": 2,
            "min_weight": 0.6,
            "focal_gamma": 3.0,
        },
        {
            "experiment_name": "5_roberta_exp02_lower_penalty",
            "experiment_description": "Lower penalty for less imbalanced dataset",
            "label_column": "basic_da",
            "model_name": "roberta-base",
            "target_modules": ["query", "key", "value", "dense"],
            "max_weight": 2,
            "min_weight": 0.6,
            "focal_gamma": 3.0,
        },
        {
            "experiment_name": "5_exp02_lower_penalty",
            "experiment_description": "Lower penalty for less imbalanced dataset",
            "label_column": "basic_da",
            "max_weight": 2,
            "min_weight": 0.6,
            "focal_gamma": 3.0,
        },
    ]
    return experiments


def run_experiment(experiment, base_params):
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: {experiment['experiment_name']}")
    print(f"Description: {experiment['experiment_description']}")
    print(f"{'='*60}")
    
    # Merge base params with experiment-specific params
    exp_params = base_params.copy()
    exp_params.update(experiment)
    
    # Print key parameters for this experiment
    print(f"Key parameters:")
    for key, value in experiment.items():
        if key != 'experiment_name' and key != 'experiment_description':
            print(f"  {key}: {value}")
    
    try:
        fine_tune_model(exp_params)
        print(f"✅ COMPLETED: {experiment['experiment_name']}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {experiment['experiment_name']}")
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False


def main():
    print("🚀 Starting Systematic Act Classification Experiments")
    print("Running 12 experiments to explore different configurations\n")
    
    base_params = get_base_params()
    experiments = get_experiments()
    
    results = {
        'completed': [],
        'failed': []
    }
    
    for i, experiment in enumerate(experiments, 1):
        print(f"\nProgress: {i}/{len(experiments)} experiments")
        
        success = run_experiment(experiment, base_params)
        
        if success:
            results['completed'].append(experiment['experiment_name'])
        else:
            results['failed'].append(experiment['experiment_name'])
            
            # Ask user if they want to continue after failure
            response = input(f"\nExperiment {experiment['experiment_name']} failed. Continue? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Final summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed: {len(results['completed'])}")
    for name in results['completed']:
        print(f"   - {name}")
    
    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}")
        for name in results['failed']:
            print(f"   - {name}")
    
    print(f"\nResults saved to: ac_results/")
    print(f"Checkpoints saved to: ac_checkpoints/")


if __name__ == "__main__":
    main()
