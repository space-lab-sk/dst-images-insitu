import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import argparse
import os


from utils import set_seed, get_torch_device, load_config, \
    count_parameters, apply_glorot_xavier, inspect_gradient_norms
from utils import Logger

from preprocessing import get_k_fold, load_data, get_torch_data
from preprocessing import StandardScaler

from postprocessing import save_gradient_norms_plot, save_predictions_and_true_values_plot, \
    save_predictions_detail_plot, save_scatter_predictions_and_true_values, \
    get_dst_rmse, get_detail_properties, get_dtw_measures



def apply_smoothing(batch_x, augumentation_rate, smoothing_window=5):

    batch_size, sequence, features = batch_x.shape
    num_samples_to_smooth = int(batch_size * augumentation_rate)

    indices_to_smooth = np.random.choice(batch_size, num_samples_to_smooth, replace=False)
    
    smoothed_batch_x = batch_x.clone()
    for idx in indices_to_smooth:
        df = pd.DataFrame(smoothed_batch_x[idx].cpu().numpy())
        smoothed_data = df.rolling(window=smoothing_window, min_periods=1).mean()
        smoothed_batch_x[idx] = torch.tensor(smoothed_data.values, device=batch_x.device, dtype=torch.float32)

    return smoothed_batch_x


def train_model(model, train_loader, optimizer, criterion, device, AUGUMENTATION_RATE):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device) , targets.to(device)
        optimizer.zero_grad()
        inputs = apply_smoothing(inputs, AUGUMENTATION_RATE)
        outputs = model(inputs)
        targets = targets.squeeze(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    mean_loss = total_loss / len(train_loader)
    return mean_loss

def validate_model(model, val_test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in val_test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.squeeze(-1))
            targets = targets.squeeze(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    mean_loss = total_loss / len(val_test_loader)
    all_outputs = torch.cat(all_outputs, dim=0)
    return mean_loss, all_outputs


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_gru_layers, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers

        self.gru = nn.GRU(input_size, hidden_size, num_gru_layers, batch_first=True, bidirectional=False, dropout=dropout)

        self.fc1 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):

        h0 = torch.zeros(self.num_gru_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)

        out = self.fc1(out[:, -1, :])

        return out
    

if __name__=="__main__":

    ########################################
    #PART 1: EXPERIMENT CONFIGURATION SETUP
    ########################################

    parser = argparse.ArgumentParser(description='Process a config file location.')
    parser.add_argument("-cfn", '--config_file_name', type=str, help='Path to the input config file')
    parser.add_argument("-dev", "--device", type=str, help="Select device: cuda:0 | cuda:1 | cpu |")
    parser.add_argument("-s", "--seed", type=str, help="Select seed")
    args = parser.parse_args()

    config_file_name = args.config_file_name
    device_input = args.device
    seed_input = args.seed
    seed_input = "42" if seed_input is None else seed_input

    # we extract all hyperparameters settings from .yamls, it was handy when we were finetuning hyperparameters
    config = load_config(f"configs/conf_gru_ie/{config_file_name}")
    EXPERIMENT_NAME = config["logging"]["experiment_name"]
    # uncomment this line to save also seed you running
    #EXPERIMENT_NAME = EXPERIMENT_NAME + "__" + seed_input
    EXPERIMENT_NOTES = config["logging"]["notes"]
    logger = Logger(EXPERIMENT_NAME)

    set_seed(int(seed_input))
    device = get_torch_device(device_input)

    logger.log_message(f"runninng seed: {seed_input}")

    BATCH_SIZE = config["training"]["batch_size"]
    LEARNING_RATE = config["training"]["learning_rate"]
    NUM_EPOCHS = config["training"]["num_epochs"]
    WEIGHT_DECAY = config["training"]["weight_decay"]
    AUGUMENTATION_RATE = config["training"]["augumentation_rate"]

    INPUT_SIZE = config["model"]["input_size"]
    HIDDEN_CHANNELS = config["model"]["hidden_channels"]
    OUTPUT_SIZE = config["model"]["output_size"]
    NUM_GRU_LAYERS = config["model"]["num_gru_layers"]
    DROPOUT = config["model"]["dropout"]

    TIME_STEPS = config["data"]["time_steps"]
    PREDICTION_WINDOW = config["data"]["prediction_window"]
    K_FOLD = config["data"]["k_fold"]


    ###########################
    #PART 2: PREPARING DATA
    ###########################

    file_ids_train, file_ids_val, file_ids_test = get_k_fold(K_FOLD)

    train_X_unscaled, train_y_unscaled = load_data(file_ids_train, time_steps=TIME_STEPS, sliding_window=PREDICTION_WINDOW)
    val_X_unscaled, val_y_unscaled = load_data(file_ids_val, time_steps=TIME_STEPS, sliding_window=PREDICTION_WINDOW)
    test_X_unscaled, test_y_unscaled = load_data(file_ids_test, time_steps=TIME_STEPS, sliding_window=PREDICTION_WINDOW)

    #droping Q1-4 data (datetime colmn was dropped in load_data()) 
    train_X_unscaled = train_X_unscaled[:, :, 4:]
    val_X_unscaled = val_X_unscaled[:, :, 4:]
    test_X_unscaled = test_X_unscaled[:, :, 4:]
    

    standard_scaler = StandardScaler(train_X_unscaled, train_y_unscaled)

    train_X = standard_scaler.standardize_X(train_X_unscaled)
    val_X = standard_scaler.standardize_X(val_X_unscaled)
    test_X =standard_scaler.standardize_X(test_X_unscaled)

    train_y = standard_scaler.standardize_y(train_y_unscaled)
    val_y = standard_scaler.standardize_y(val_y_unscaled)
    test_y = standard_scaler.standardize_y(test_y_unscaled)

    train_X, train_y = get_torch_data(train_X, train_y)
    val_X, val_y = get_torch_data(val_X, val_y)
    test_X, test_y = get_torch_data(test_X, test_y)


    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###############################################
    #PART 3: DEEP LEARNING PART
    ###############################################

    model = GRUModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_CHANNELS, output_size=OUTPUT_SIZE, num_gru_layers=NUM_GRU_LAYERS, dropout=DROPOUT)
    model.to(device)
    apply_glorot_xavier(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    print(model)
    logger.log_message(model)

    total_params = count_parameters(model)
    print(f"Total parameters of the model: {total_params}")
    logger.log_message(f"Total parameters of the model: {total_params}")


    ############################
    #PART 4: TRAINING LOOP
    ############################

    print(f"--------------TRAINING LOOP--------------")
    logger.log_message(f"--------------TRAINING LOOP--------------")
    
    losses = []
    val_losses = []
    gradient_norms = []

    best_val = 10000.0
    best_model_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, criterion, device, AUGUMENTATION_RATE)
        val_loss, _ = validate_model(model, val_loader, criterion, device)

        losses.append(train_loss)
        val_losses.append(val_loss)

        #==============grad norms============
        total_norm = inspect_gradient_norms(model)
        gradient_norms.append(total_norm)
        #==========================

        print(f'{epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        logger.log_message(f'{epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        if val_loss < best_val:
            best_val = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"model with val loss {val_loss} saved...")
            logger.log_message(f"model with val loss {val_loss} saved...")

    print('Training completed saving....')
    logger.log_message('Training completed saving....')
    torch.save(best_model_state, f'models/{EXPERIMENT_NAME}.pth')

    save_gradient_norms_plot(gradient_norms, save_path=f"logs/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_grad_norms.png")
    

    ############################
    #PART 4: MODEL EVALUATION
    ############################

    model.load_state_dict(best_model_state)
    

    test_loss, test_predictions_standardized = validate_model(model, test_loader, criterion, device)
    test_predictions_standardized = test_predictions_standardized.cpu()
    test_predictions = (test_predictions_standardized * standard_scaler.y_std) + standard_scaler.y_mean
    print(test_predictions_standardized.shape)
    test_predictions = test_predictions.numpy().tolist()
    print(f"avg. test loss {test_loss}")
    logger.log_message(f"avg. test loss {test_loss}")
    
    #test_y_np = test_y_unscaled
    #test_y_np = np.squeeze(test_y_np, -1)
    y_true_list = test_y_unscaled.tolist()
    print(test_y_unscaled.shape)

    save_predictions_and_true_values_plot(y_true_list, 
                                          test_predictions, 
                                          save_path=f"logs/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_targets_and_preds.png")
    
    # plot in detail 3 geomagnetic storms period from test set for different k-folds
    for detail_number in range (3):
        detail_start, detail_end, detail_name = get_detail_properties(K_FOLD, detail=detail_number)

        save_predictions_detail_plot(y_true_list, 
                                    test_predictions, 
                                    save_path=f"logs/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_detail_{detail_number}.png",
                                    detail_start=detail_start,
                                    detail_end=detail_end,
                                    detail_name=detail_name)
    
    
    save_scatter_predictions_and_true_values(y_true_list, 
                                             test_predictions, 
                                             save_path=f"logs/{EXPERIMENT_NAME}/{EXPERIMENT_NAME}_targets_and_preds_scatter.png")
    
    
    print(test_y_unscaled.shape)
    print(np.array(test_predictions).shape)

    dst_rmse = get_dst_rmse(test_y_unscaled, test_predictions)
    on_diagonal_percentage, diagonal_sum = get_dtw_measures(test_y_unscaled, test_predictions, PREDICTION_WINDOW)

    print(f"Dst RMSE on test set between targets and predictions: {dst_rmse:.5f}")
    logger.log_message(f"Dst RMSE on test set between targets and predictions: {dst_rmse:.5f}")

    print(f"DTW: steps on diagonal percentage: {on_diagonal_percentage}")
    logger.log_message(f"DTW: steps on diagonal percentage: {on_diagonal_percentage}")

    print(f"DTW: diagonal sum values: {diagonal_sum}")
    logger.log_message(f"DTW: diagonal sum values: {diagonal_sum}")

    print(f"lowest val. loss: {best_val:.5f}")
    logger.log_message(f"lowest val. loss: {best_val:.5f}")

