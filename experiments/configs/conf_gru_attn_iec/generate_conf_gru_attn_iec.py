import yaml

def generate_config(batch_num, param_to_change, param_values, k_folds, base_file_name):
    config_template = {
        'logging': {
            'experiment_name': '',
            'notes': f'Experiment from batch {batch_num}'
        },
        'model': {
            'input_size': 9,
            'output_size': 1,
            'num_gru_layers': 2,
            'hidden_channels1': 256,
            'hidden_channels2': 256,
            'dropout': 0.0
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.00003,
            'weight_decay': 0.0001,
            'augumentation_rate': 0.0
        },
        'data': {
            'time_steps': 200,
            'prediction_window': 10,
            'k_fold': 1
        }
    }

    for idx, value in enumerate(param_values):
        for k_fold in k_folds:
    
            experiment_name = f'{batch_num}_{idx + 1}_{k_fold}'
            config_template['logging']['experiment_name'] = f'gru_attn_iec_{experiment_name}'
            config_template['data']['k_fold'] = k_fold
            
            keys = param_to_change.split('.')
            config_section = config_template
            for key in keys[:-1]:
                config_section = config_section[key]
            config_section[keys[-1]] = value         
            
            file_name = f"{base_file_name}_{experiment_name}.yaml"
            with open(file_name, 'w') as file:
                yaml.dump(config_template, file, default_flow_style=False)
            
            print(f"Generated config: {file_name}")


param_to_change = 'data.prediction_window'
param_values = [10, 20]
experiment_batch = 1
generate_config(batch_num=experiment_batch, param_to_change=param_to_change, param_values=param_values, k_folds=[1,2,3,4,5], base_file_name='conf_gru_attn_iec')
