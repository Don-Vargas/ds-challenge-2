from collections import defaultdict

def dataset_building(datasets, all_rankings, y, role='train'):
    '''
    :param datasets: dict of pandas DataFrames
    :param all_rankings: dict with feature rankings
    :param y: pandas Series or array-like target
    :param role: 'train', 'test', or other (e.g. 'inference')
    '''
    ds = defaultdict(dict)
    
    for dataset_name, ranking_info in all_rankings.items():
        top_features = list(ranking_info['top_features'])
        data_frame = datasets[dataset_name].copy()
        all_bool = all(data_frame[col].dtype == bool for col in data_frame.columns)
        missing_cols = set(top_features) - set(data_frame.columns)

        print(dataset_name)
        print(f'missing columns: \n{missing_cols}') if missing_cols else print('no missing columns')
        fill_value = False if all_bool else 0
        for col in missing_cols:
            data_frame[col] = fill_value

        data_frame = data_frame[top_features]

        # Add y only for train/test
        if role in ('train', 'test'):
            data_frame['target'] = y

        ds[dataset_name] = data_frame

    for dataset_name in ['ds7', 'ds8', 'ds9', 'ds10']:
        data_frame = datasets[dataset_name].copy()

        if role in ('train', 'test'):
            data_frame['target'] = y

        ds[dataset_name] = data_frame

    return ds
