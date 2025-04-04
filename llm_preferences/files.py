import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

file_dict = {
    "Anthropic/hh-rlhf": {
        # os.path.join(THIS_DIR,
        #              'LLM_rewards_model_no_NA-Mistral_data_trained-'
        #              'NaN_dataset_eval-hh-harmlesness.csv'): 'Mistral-7B (base) w/o NA',
        # os.path.join(THIS_DIR,
        #              'LLM_rewards_model_no_NA-Mistral_data_trained-hh-'
        #              'harmlesness_dataset_eval-hh-harmlesness.csv'): 'Mistral-7B (finetuned) w/o NA',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Mistral_data_trained-'
                     'NaN_dataset_eval-hh-harmlesness.csv'): 'Mistral-7B (base)',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Mistral_data_trained-hh-'
                     'harmlesness_dataset_eval-hh-harmlesness.csv'): 'Mistral-7B (finetuned)',
        # os.path.join(THIS_DIR,
        #              'LLM_rewards_model_no_NA-Llama3-8B_data_trained-'
        #              'NaN_dataset_eval-hh-harmlesness.csv'): 'Llama3-8B (base) w/o NA',
        # os.path.join(THIS_DIR,
        #              'LLM_rewards_model_no_NA-Llama3-8B_data_trained-hh-'
        #              'harmlesness_dataset_eval-hh-harmlesness.csv'): 'Llama3-8B (finetuned) w/o NA',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Llama3-8B_data_trained-'
                     'NaN_dataset_eval-hh-harmlesness.csv'): 'Llama3-8B (base)',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Llama3-8B_data_trained-'
                     'hh-harmlesness_dataset_eval-hh-harmlesness.csv'): 'Llama3-8B (finetuned)',
        os.path.join(THIS_DIR,
                     'hh_test_generic_4ob.csv'): 'GPT-4o',
        os.path.join(THIS_DIR,
                     'hh_test_generic_deepseekv3.csv'): 'DeepSeek-V3',
        os.path.join(THIS_DIR,
                'hh_test_generic_sonnet3.5.csv'): 'Sonnet3.5',
    },
    "PKU-Alignment/PKU-SafeRLHF": {
        # os.path.join(THIS_DIR,
        #              'LLM_rewards_model_no_NA-Mistral_data_trained-'
        #              'NaN_dataset_eval-pku_splited.csv'): 'Mistral-7B (base) w/o NA',
        # os.path.join(THIS_DIR,
        #              'LLM_rewards_model_no_NA-Mistral_data_trained-'
        #              'pku_single_dataset_eval-pku_splited.csv'): 'Mistral-7B (finetuned) w/o NA',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Mistral_data_trained-'
                     'NaN_dataset_eval-pku_splited.csv'): 'Mistral-7B (base)',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Mistral_data_trained-'
                     'pku_single_dataset_eval-pku_splited.csv'): 'Mistral-7B (finetuned)',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Llama3-8B_data_trained-'
                     'NaN_dataset_eval-pku_splited.csv'): 'Llama3-8B (base)',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Llama3-8B_data_trained-'
                     'pku_single_dataset_eval-pku_splited.csv'): 'Llama3-8B (finetuned)',
        os.path.join(THIS_DIR,
                     'pku_splited_test_generic_4ob.csv'): 'GPT-4o',
        os.path.join(THIS_DIR,
                     'pku_splited_test_generic_deepseekv3.csv'): 'DeepSeek-V3',
        os.path.join(THIS_DIR,
                'pku_splited_test_generic_sonnet3.5.csv'): 'Sonnet3.5',
},
    "PKU-Alignment/PKU-SafeRLHF-single-dimension": {
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Mistral_data_trained-'
                     'NaN_dataset_eval-pku_single.csv'): 'Mistral-7B (base)',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Mistral_data_trained-'
                     'pku_single_dataset_eval-pku_single.csv'): 'Mistral-7B (finetuned)',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Llama3-8B_data_trained-'
                     'NaN_dataset_eval-pku_single.csv'): 'Llama3-8B (base)',
        os.path.join(THIS_DIR,
                     'LLM_rewards_model_with_NA-Llama3-8B_data_trained-'
                     'pku_single_dataset_eval-pku_single.csv'): 'Llama3-8B (finetuned)',
        os.path.join(THIS_DIR,
                     'pku_single_test_generic_4ob.csv'): 'GPT-4o',
        os.path.join(THIS_DIR,
                     'pku_single_test_generic_deepseekv3.csv'): 'DeepSeek-V3',
        os.path.join(THIS_DIR,
        'pku_single_test_generic_sonnet3.5.csv'): 'Sonnet3.5',
    }
}
