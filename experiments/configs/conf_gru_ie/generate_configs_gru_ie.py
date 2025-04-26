import yaml

def generate_config(batch_num, params_to_change, param_values, k_folds, base_file_name):
    config_template = {
        'logging': {
            'experiment_name': '',
            'notes': f'Experiment from batch {batch_num}'
        },
        'model': {
            'input_size': 9,
            'output_size': 1,
            'num_gru_layers': 2,
            'hidden_channels': 64,
            'dropout': 0.2
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 200,
            'learning_rate': 0.00001,
            'weight_decay': 0.1,
            'augumentation_rate': 0.1
        },
        'data': {
            'time_steps': 100,
            'prediction_window': 10,
            'k_fold': 1
        }
    }

    # Ensure all parameter value lists are of the same length
    num_configs = len(param_values[0])
    assert all(len(values) == num_configs for values in param_values), "All parameter value lists must have the same length."

    for idx in range(num_configs):
        for k_fold in k_folds:
            experiment_name = f'{batch_num}_{idx + 1}_{k_fold}'
            config_template['logging']['experiment_name'] = f'gru_ie_{experiment_name}'
            config_template['data']['k_fold'] = k_fold
            
            # Update configuration for the current parameter index
            for param, values in zip(params_to_change, param_values):
                keys = param.split('.')
                config_section = config_template
                for key in keys[:-1]:
                    config_section = config_section[key]
                config_section[keys[-1]] = values[idx]
            
            # Save the configuration to a file
            file_name = f"{base_file_name}_{experiment_name}.yaml"
            with open(file_name, 'w') as file:
                yaml.dump(config_template, file, default_flow_style=False)
            
            print(f"Generated config: {file_name}")


params_to_change = ['data.prediction_window']
param_values = [[10, 20]]
experiment_batch = 1
generate_config(batch_num=experiment_batch, params_to_change=params_to_change, param_values=param_values, k_folds=[1,2,3,4,5], base_file_name='conf_gru_ie')

