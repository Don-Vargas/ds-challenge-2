from collections import defaultdict

def dataset_building(datasets, all_rankings):
    '''
    Docstring for dataset_building
    
    :param datasets: Description
    :param all_rankings: Description
    '''
    ds = defaultdict(dict)
    
    for dataset_name, ranking_info in all_rankings.items():
        top_features = list(ranking_info['top_features'])
        data_frame = datasets[dataset_name].copy()
        all_bool = all(data_frame[col].dtype == bool for col in data_frame.columns)
        missing_cols = set(top_features) - set(data_frame.columns)

        print(f'missing columns: n\ {missing_cols}') if missing_cols else print('no missing columns')
        fill_value = False if all_bool else 0
        for col in missing_cols:
            data_frame[col] = fill_value

        ds[dataset_name] = data_frame[top_features]

    for dataset_name in ['ds7','ds8','ds9','ds10']:
        ds[dataset_name] = datasets[dataset_name]

    return ds
