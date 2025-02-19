import torch
import torch.nn as nn
import numpy as np
import copy
import argparse

from utils import set_seed, get_torch_device, load_config, \
    count_parameters, apply_glorot_xavier, inspect_gradient_norms
from utils import Logger

from preprocessing import get_k_fold, load_data, get_torch_data
from preprocessing import StandardScaler

from postprocessing import save_gradient_norms_plot, save_predictions_and_true_values_plot, \
    save_predictions_detail_plot, save_scatter_predictions_and_true_values, \
    get_dst_rmse, get_detail_properties, get_dtw_measures


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        # ----split coronagraph data (first four columns) and other data
        quadrands_inputs = inputs[:, :, :4].to(device)
        inputs = inputs[:, :, 4:].to(device)
        #---------------
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, quadrands_inputs)
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
            quadrands_inputs = inputs[:, :, :4].to(device)
            inputs = inputs[:, :, 4:].to(device)
            targets = targets.to(device)
            outputs = model(inputs, quadrands_inputs)
            all_outputs.append(outputs.squeeze(-1))
            targets = targets.squeeze(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    mean_loss = total_loss / len(val_test_loader)
    all_outputs = torch.cat(all_outputs, dim=0)
    return mean_loss, all_outputs


class QKVAttention(nn.Module):
    """Multi-Head Scaled Dot-Product Attention"""
    def __init__(self, hidden_size, num_heads):
        super(QKVAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        """
        query: [batch_size, 1, hidden_size]  (previous hidden state)
        key: [batch_size, seq_len, hidden_size] (GRU outputs)
        value: [batch_size, seq_len, hidden_size] (GRU outputs)
        """
        attn_output, attn_weights = self.mha(query, key, value)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_size]
        
        return attn_output, attn_weights
    

class GRUWithQKVAttention(nn.Module):
    """GRU layer followed by QKV Self-Attention"""
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(GRUWithQKVAttention, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
        self.attention = QKVAttention(hidden_size, num_heads)

    def forward(self, x, h):
        output, _ = self.gru(x, h)  # [batch_size, seq_len, hidden_size]
        context, attention_weights = self.attention(output[:, -1:, :], output, output)
        return context, attention_weights
    

class GRUModelWithAttention(nn.Module):
    """Complete Model with GRU + Multi-Head QKV Attention"""
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, num_gru_layers, dropout, num_heads):
        super(GRUModelWithAttention, self).__init__()
        
        self.num_gru_layers = num_gru_layers
        
        self.hidden_size = hidden_size
        self.gru1 = nn.GRU(input_size, hidden_size, num_gru_layers, batch_first=True, bidirectional=False, dropout=0.0)
        
        # found that mapping coronagraph visual features (Q-data) to 16-dim works best
        self.linear1 = nn.Linear(4, 16)
        
        self.hidden_size2 = hidden_size2
        self.gru2 = GRUWithQKVAttention(16, self.hidden_size2, num_gru_layers, num_heads)
        
        self.fc_out = nn.Linear(hidden_size + hidden_size2, output_size)
        self.gelu = nn.GELU()

    def forward(self, x, x_q):
        h0 = torch.zeros(self.num_gru_layers, x.size(0), self.hidden_size).to(x.device)
        out1, _ = self.gru1(x, h0)

        x_q = self.gelu(self.linear1(x_q))

        h02 = torch.zeros(self.num_gru_layers, x_q.size(0), self.hidden_size2).to(x.device)
        context, weights = self.gru2(x_q, h02)

        concated = torch.cat((out1[:, -1, :], context), 1)
        out = self.fc_out(concated)
        return out


def bin_features(X, feature_indices, bins):
    X_binned = X.copy()
    for idx in feature_indices:
        X_binned[..., idx] = np.digitize(X[..., idx], bins=bins, right=False)
    return X_binned


def separate_features(data, binned_indices, scaled_indices):
    binned_features = data[..., binned_indices]
    scaled_features = data[..., scaled_indices]
    return binned_features, scaled_features


def combine_features(binned, scaled):
    return np.concatenate([binned, scaled], axis=-1)


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

    config = load_config(f"configs/conf_gru_attn_iec/{config_file_name}")
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
    HIDDEN_CHANNELS1 = config["model"]["hidden_channels1"]
    HIDDEN_CHANNELS2 = config["model"]["hidden_channels2"]
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

    #standard_scaler = StandardScaler(train_X_unscaled, train_y_unscaled)

    #--------conver coronagraph visual features (Q-DATA) from number to category -- categorization------------
    # - similar as savgol filter in preprocessing.load_data() categorization can be left out, but we figured out that it gives small performance boost, so we kept it here

    standard_scaler = StandardScaler(train_X_unscaled[..., 4:], train_y_unscaled)

    bin_edges = [0, 3.0, 5.0, 15.0, 30.0, 100.0]  
    features_to_bin = [0, 1, 2, 3]
    features_to_scale = range(4, train_X_unscaled.shape[-1])

    # separate binned q-data and scaled features
    train_binned, train_to_scale = separate_features(train_X_unscaled, range(4), features_to_scale)
    val_binned, val_to_scale = separate_features(val_X_unscaled, range(4), features_to_scale)
    test_binned, test_to_scale = separate_features(test_X_unscaled, range(4), features_to_scale)

    # standardize only the scaled features
    train_scaled = standard_scaler.standardize_X(train_to_scale)
    val_scaled = standard_scaler.standardize_X(val_to_scale)
    test_scaled = standard_scaler.standardize_X(test_to_scale)

    train_X_binned = bin_features(train_binned, features_to_bin, bin_edges) 
    val_X_binned = bin_features(val_binned, features_to_bin, bin_edges) 
    test_X_binned = bin_features(test_binned, features_to_bin, bin_edges) 
    
    # combine bined q-data and standardized numeric insitu
    train_X = combine_features(train_X_binned, train_scaled)
    val_X = combine_features(val_X_binned, val_scaled)
    test_X = combine_features(test_X_binned, test_scaled)

    #-----------------------------------------

    #train_X = standard_scaler.standardize_X(train_X_unscaled)
    #val_X = standard_scaler.standardize_X(val_X_unscaled)
    #test_X =standard_scaler.standardize_X(test_X_unscaled)

    # standardize y
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

    model = GRUModelWithAttention(input_size=INPUT_SIZE, hidden_size=HIDDEN_CHANNELS1, hidden_size2=HIDDEN_CHANNELS2, output_size=OUTPUT_SIZE, num_gru_layers=NUM_GRU_LAYERS, dropout=DROPOUT, num_heads=2)
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
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
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
    test_predictions = test_predictions.numpy().tolist()
    print(f"avg. test loss {test_loss}")
    logger.log_message(f"avg. test loss {test_loss}")
    
    #test_y_np = test_y_unscaled
    #test_y_np = np.squeeze(test_y_np, -1)
    y_true_list = test_y_unscaled.tolist()

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
    
    
    save_scatter_predictions_and_true_values(test_y_unscaled, 
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
