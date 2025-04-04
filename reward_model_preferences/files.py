import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

file_dict = {
    "Anthropic/hh-rlhf": {
        os.path.join(THIS_DIR, 'Mistral7B_reward_model_trained_in-hh-harmlesness_evaluated_in-hh'
                               '-harmlesness_partition-test.csv'): 'Mistral-7B',
        os.path.join(THIS_DIR, 'Llama3-8B_reward_model_trained_in-hh-harmlesness_evaluated_in-hh'
                               '-harmlesness_partition-test.csv'): 'Llama3-8B',
        os.path.join(THIS_DIR, 'reward_preference_data_reward_model-most_downloaded_hh'
                               '_in_dataset_hh_partition-None.csv'): 'Most downloaded',
    },
    "PKU-Alignment/PKU-SafeRLHF": {
        os.path.join(THIS_DIR,
                     'Mistral7B_reward_model_trained_in-pku'
                     '_evaluated_in-pku_splited_partition-test.csv'): 'Mistral-7B',
        os.path.join(THIS_DIR,
                     'Llama3-8B_reward_model_trained_in-pku'
                     '_evaluated_in-pku_splited_partition-test.csv'): 'Llama3-8B',
        os.path.join(THIS_DIR,
                     'reward_preference_data_reward_model-most_downloaded_pku'
                     '_in_dataset_pku_splited_partition-None.csv'): 'Most downloaded',
    },
    "PKU-Alignment/PKU-SafeRLHF-single-dimension": {
        os.path.join(THIS_DIR,
                     'Mistral7B_reward_model_trained_in-pku'
                     '_evaluated_in-pku_partition-test.csv'): 'Mistral-7B',
        os.path.join(THIS_DIR,
                     'Llama3-8B_reward_model_trained_in-pku'
                     '_evaluated_in-pku_partition-test.csv'): 'Llama3-8B',
        os.path.join(THIS_DIR,
                     'reward_preference_data_reward_model-most_downloaded_pku'
                     '_in_dataset_pku_partition-None.csv'): 'Most downloaded',
    }
}
