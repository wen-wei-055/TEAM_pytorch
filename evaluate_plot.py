import argparse
import os
import numpy as np
import json
import pickle
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import h5py
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import torch

import models
from models import EnsembleEvaluateModel
import loader
import util
import plots

from util import generator_from_config

EARTH_RADIUS = 6371

sns.set(font_scale=1.5)
sns.set_style('ticks')

def predict_at_time(model, time, data, event_metadata, batch_size, config, sampling_rate=100, pga=False,
                    use_multiprocessing=True, no_event_token=False, dataset_id=None):
    generator = generator_from_config(config, data, event_metadata, time, batch_size, sampling_rate, dataset_id=dataset_id)

    workers = 1
    if use_multiprocessing:
        workers = 10
    predictions, labels = model.prediction_generator_partial_station(generator, workers=workers, use_multiprocessing=use_multiprocessing)
    return predictions, labels


def generate_true_pred_plot(pred_values, true_values, time, path, suffix=''):
    if suffix:
        suffix += '_'
        
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    _, cbar = plots.true_predicted(true_values, pred_values, agg='mean', quantile=True, ax=ax, time=time)
    fig.savefig(os.path.join(path, f'truepred_{suffix}{time}.png'), bbox_inches='tight')
    plt.close(fig)


def generate_calibration_plot(pred_values, true_values, time, path, suffix=''):
    if suffix:
        suffix += '_'
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    plots.calibration_plot(pred_values, true_values, ax=ax)
    ax.set_xlabel('<-- Overestimate       Underestimate -->')
    fig.savefig(os.path.join(path, f'quantiles_{suffix}{time}.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--weight_file', type=str)  # If unset use latest model
    parser.add_argument('--times', type=str, default='0.5,1,2,4,8,16,25')  # Has only performance implications
    parser.add_argument('--max_stations', type=int)  # Overwrite max stations value from config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val', action='store_true')  # Evaluate on val set
    parser.add_argument('--n_pga_targets', type=int)  # Overwrite number of PGA targets
    parser.add_argument('--head_times', action='store_true')  # Evaluate warning times
    parser.add_argument('--blind_time', type=float, default=0.5)  # Time of first evaluation after first P arrival
    parser.add_argument('--alpha', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')  # 機率大於alpha以上就會發布警報 Probability thresholds alpha
    parser.add_argument('--additional_data', type=str)  # Additional data set to use for evaluation
    parser.add_argument('--dataset_id', type=int)  # ID of dataset to evaluate on, in case of joint training
    parser.add_argument('--wait_file', type=str)  # Wait for this file to exist before starting evaluation
    parser.add_argument('--ensemble_member', action='store_true')  # Task to evaluate is an ensemble member
                                                                   # (not the full ensembel)
    parser.add_argument('--loss_limit', type=float) # In ensemble model, discard members with loss above this limit
    parser.add_argument('--no_multiprocessing', action='store_true')
    args = parser.parse_args()

    if args.wait_file is not None:
        util.wait_for_file(args.wait_file)

    times = [float(x) for x in args.times.split(',')]

    config = json.load(open(os.path.join(args.experiment_path, 'config.json'), 'r'))
    training_params = config['training_params']

    
    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")
    
    
    if (args.dataset_id is None) and (isinstance(training_params['data_path'], list) and
                                      len(training_params['data_path']) > 1):
        raise ValueError('dataset_id needs to be set for experiments with multiple input data sets.')
    if (args.dataset_id is not None) and not (isinstance(training_params['data_path'], list) and
                                              len(training_params['data_path']) > 1):
        raise ValueError('dataset_id may only be set for experiments with multiple input data sets.')

    if args.dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[args.dataset_id]
        data_path = training_params['data_path'][args.dataset_id]
        n_datasets = len(training_params['data_path'])
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]
        data_path = training_params['data_path']
        n_datasets = 1
        
    
    key = generator_params.get('key', 'MA')
    pos_offset = generator_params.get('pos_offset', (-21, -69))
    pga_key = generator_params.get('pga_key', 'pga')

    if args.blind_time != 0.5:
        suffix = f'_blind{args.blind_time:.1f}'
    else:
        suffix = ''

    if args.dataset_id is not None:
        suffix += f'_{args.dataset_id}'

    if args.val:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'val')
        training_params['data_path'] = ""
        test_set = False
    else:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'test')
        training_params['data_path'] = ""
        test_set = True

    if not os.path.isdir(os.path.join(args.experiment_path, f'evaluation{suffix}')):
        os.mkdir(os.path.join(args.experiment_path, f'evaluation{suffix}'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    batch_size = generator_params['batch_size']
    shuffle_train_dev = generator_params.get('shuffle_train_dev', False)
    custom_split = generator_params.get('custom_split', None)
    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_mag = generator_params.get('min_mag', None)
    mag_key = generator_params.get('key', 'MA')
    event_metadata, data, metadata = loader.load_events(training_params['data_path'],
                                                        shuffle_train_dev=shuffle_train_dev,
                                                        custom_split=custom_split,
                                                        min_mag=min_mag,
                                                        mag_key=mag_key,
                                                        overwrite_sampling_rate=overwrite_sampling_rate)

    if args.additional_data:
        print('Loading additional data')
        event_metadata_add, data_add, _ = loader.load_events(args.additional_data,
                                                             parts=(True, True, True),
                                                             min_mag=min_mag,
                                                             mag_key=mag_key,
                                                             overwrite_sampling_rate=overwrite_sampling_rate)
        event_metadata = pd.concat([event_metadata, event_metadata_add])
        for t_key in data.keys():
            if t_key in data_add:
                data[t_key] += data_add[t_key]

    if 'max_stations' not in config['model_params']:
        config['model_params']['max_stations'] = data['waveforms'].shape[1]
    if args.max_stations is not None:
        config['model_params']['max_stations'] = args.max_stations
        
    args.n_pga_targets = 249
    
    if args.n_pga_targets is not None:
        if config['model_params'].get('n_pga_targets', 0) > 0:
            print('Overwriting number of PGA targets')
            config['model_params']['n_pga_targets'] = args.n_pga_targets
        else:
            print('PGA flag is set, but model does not support PGA')

    
    ensemble = config.get('ensemble', 1)
    print(ensemble)
    if ensemble > 1 and not args.ensemble_member:
        experiment_path = args.experiment_path
        model_list = EnsembleEvaluateModel(config, experiment_path, loss_limit=args.loss_limit, batch_size=batch_size, device=device)

    else: 
        if 'n_datasets' in config['model_params']:
            del config['model_params']['n_datasets']
        model_list = models.build_transformer_model(**config['model_params'], trace_length=data['waveforms'][0].shape[1], n_datasets=n_datasets).to(device)

        if args.weight_file is not None:
            weight_file = os.path.join(args.experiment_path, args.weight_file)
        else:
            weight_file = sorted([x for x in os.listdir(args.experiment_path) if x[:11] == 'checkpoint-'])[-1]
            weight_file = os.path.join(args.experiment_path, weight_file)
        
        model_list.load_state_dict(torch.load(weight_file)['model_weights'])

    mag_stats = []
    loc_stats = []
    pga_stats = []
    mag_pred_full = []
    loc_pred_full = []
    pga_pred_full = []
    
    for time in times:
        print(f'Time: {time} s')
        pga_pred, pga_true = predict_at_time(model_list, time, data, event_metadata,
                               config=config,
                               batch_size=batch_size,
                               sampling_rate=metadata['sampling_rate'],
                               dataset_id=args.dataset_id)
        pga_pred_reshaped = pga_pred
        pga_true_reshaped = pga_true
        generate_true_pred_plot(pga_pred_reshaped, pga_true_reshaped, time, output_dir, suffix='pga')
        generate_calibration_plot(pga_pred_reshaped, pga_true_reshaped, time, output_dir, suffix='pga')
    
    results = {'times': times,
               'pga_stats': np.array(pga_stats).tolist()}

    with open(os.path.join(output_dir, 'stats.json'), 'w') as stats_file:
        json.dump(results, stats_file, indent=4)

    if args.head_times:
        times_pga = np.arange(args.blind_time, 25, 0.2)
        alpha = [float(x) for x in args.alpha.split(',')]
        warning_time_information = calculate_warning_times(config, model_list, data, event_metadata,
                                                           times=times_pga,
                                                           alpha=alpha,
                                                           batch_size=batch_size,
                                                           use_multiprocessing=not args.no_multiprocessing,
                                                           dataset_id=args.dataset_id, device=device)
    else:
        warning_time_information = None
        alpha = None


    with open(os.path.join(output_dir, 'predictions_all.pkl'), 'wb') as pred_file:
        pickle.dump((times, mag_pred_full, loc_pred_full, pga_pred_full, warning_time_information, alpha), pred_file)
        