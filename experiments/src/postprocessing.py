import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from dtw_measure import dtw_measure

def get_dtw_measures(test_y_unscaled, test_predictions, PREDICTION_WINDOW):
    M, path, cost = dtw_measure(test_y_unscaled, test_predictions, PREDICTION_WINDOW)
    # if using large timeseries (above 1000) cost matrix M is 
    # scipy sparse matrix <class 'scipy.sparse.csr.csr_matrix'>
    # if using less, M is regular numpy array
    # in original implementation, M is not returned when 
    # large series are provided, we changed that for our purpouse
    
    bins, counts = np.unique(abs(path[0, :] - path[1, :]), return_counts=True)
    on_diagonal_count = counts[0]
    non_on_diagonal_count = sum(counts) - counts[0]
    on_diagonal_percentage = counts[0] / sum(counts)
    diagonal_sum = sum(M[i, i] for i in range(M.shape[0]))

    
    #print(f"on diagonal count: {on_diagonal_count}")
    #print(f"non diagonal count: {non_on_diagonal_count}")
    #print(f"on diagonal percentage: {on_diagonal_percentage}")
    #print(f"{M.shape}")
    #print(f"{M.ndim}")
    #print(type(M))
    #print(f"diagonal sum M: {diagonal_sum}")

    return on_diagonal_percentage, diagonal_sum


def get_dst_rmse(test_y, predictions):

    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    if isinstance(test_y, list):
        test_y = np.array(test_y)

    rmse = np.sqrt(np.mean(np.square(test_y-predictions)))
    return rmse


def get_r_squared(test_y: np.ndarray, predictions: np.ndarray):

    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    if isinstance(test_y, list):
        test_y = np.array(test_y)

    if predictions.ndim > 1:
        predictions = predictions.reshape(-1)

    if test_y.ndim > 1:
        test_y = test_y.reshape(-1)

    mean_true_values = np.mean(test_y)
    sst = np.sum((test_y - mean_true_values) ** 2)
    ssr = np.sum((test_y - predictions) ** 2)

    r_squared = 1 - (ssr / sst)
    return r_squared


def save_gradient_norms_plot(gradient_norms, save_path:str):
    plt.plot(gradient_norms)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm over Time')
    plt.savefig(save_path)


def save_predictions_and_true_values_plot(y_true, predictions, save_path:str):
    plt.figure(figsize=(20, 5))
    plt.plot(y_true, label="True values", linewidth=1, color="green")
    plt.plot(predictions, label="Prediction", color='orange', linewidth=1)

    plt.legend()
    plt.xlabel('index')
    plt.ylabel('Values')
    plt.grid(True)
    #plt.show()
    plt.savefig(save_path)


def save_predictions_detail_plot(y_true, 
                                 predictions,  
                                 save_path:str, 
                                 detail_start: int, 
                                 detail_end: int,
                                 detail_name: str):
    
    plt.figure(figsize=(20, 5))
    plt.plot(y_true[detail_start:detail_end], label="True values", linewidth=0.5, color="green", marker='o', markersize=3)
    plt.plot(predictions[detail_start:detail_end], label="Prediction", linewidth=0.5, color="orange", marker='o', markersize=3)
    plt.legend()
    plt.xlabel('index')
    plt.ylabel('Values')
    plt.title(detail_name)
    plt.grid(True)
    #plt.show()
    plt.savefig(save_path)


def save_scatter_predictions_and_true_values(test_y: np.ndarray, predictions: np.ndarray, save_path:str):

    if isinstance(test_y, list):
        test_y = np.array(test_y)
    
    if isinstance(predictions, list):
        predictions = np.array(predictions)

    test_y = test_y.flatten()
    predictions = predictions.flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(test_y, predictions, label="Predictions", alpha=0.5, color="orange")
    min_val = min(test_y.min(), predictions.min())
    max_val = max(test_y.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', label="Ideal")
    plt.xlabel("True Values [nT]")
    plt.ylabel("Predicted Values [nT]")
    plt.title("Predicted and True Values Scatter")
    plt.legend()
    #plt.show()
    plt.savefig(save_path)
    

def get_detail_properties(K_FOLD: int, detail: int, longer_context: bool=False):
    """ gets detail properties for graph. available K_folds = {1,..,5} ; available details = {1,..,3}"""
    
    padding = 200 if longer_context else 0
    
    k_folds_data = {
    1: [
        {"title": "Detail on event 38", "detail_start": 880 - int((padding/2)), "detail_end": 1050 - int((padding/2))},
        {"title": "Detail on event 15", "detail_start": 2450 - padding, "detail_end": 2600 - padding},
        {"title": "Detail on event 23", "detail_start": 3670 - padding, "detail_end": 3950 - padding}
    ],
    2: [
        {"title": "Detail on event 34", "detail_start": 1520 - padding, "detail_end": 1700 - padding},
        {"title": "Detail on event 17", "detail_start": 2520 - padding, "detail_end": 2750 - padding},
        {"title": "Detail on event 11", "detail_start": 3770 - padding, "detail_end": 3950 - padding}
    ],
    3: [
        #{"title": "Detail on event 35", "detail_start": 1000 - padding, "detail_end": 1200 - padding},
        {"title": "Detail on event 35", "detail_start": 900 - padding, "detail_end": 1100 - padding},
        {"title": "Detail on event 14", "detail_start": 2520 - padding, "detail_end": 2750 - padding},
        {"title": "Detail on event 23", "detail_start": 3770 - padding, "detail_end": 3950 - padding}
    ],
    4: [
        {"title": "Detail on event 34", "detail_start": 1520 - padding, "detail_end": 1700 - padding},
        {"title": "Detail on event 15", "detail_start": 2400 - padding, "detail_end": 2520 - padding},
        {"title": "Detail on event 11", "detail_start": 3470 - padding, "detail_end": 3950 - padding}
    ],
    5: [
        {"title": "Detail on event 8", "detail_start": 500 - padding, "detail_end": 600 - padding},
        {"title": "Detail on event 30", "detail_start": 1250 - padding, "detail_end": 1420 - padding},
        {"title": "Detail on event 12", "detail_start": 2770 - padding, "detail_end": 2950 - padding}
    ],
    6: [
        {"title": "Detail on event 9", "detail_start": 500 - padding, "detail_end": 700 - padding},
        {"title": "Detail on event 22", "detail_start": 850 - padding, "detail_end": 1200 - padding}
    ]
}
    
    detail_start = k_folds_data[K_FOLD][detail]["detail_start"]
    detail_end = k_folds_data[K_FOLD][detail]["detail_end"]
    detail_name = k_folds_data[K_FOLD][detail]["title"]

    return detail_start, detail_end, detail_name


