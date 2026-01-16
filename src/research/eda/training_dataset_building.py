from collections import defaultdict

def dataset_building(datasets, all_rankings):
    '''
    Docstring for dataset_building
    
    :param datasets: Description
    :param all_rankings: Description
    '''
    ds = defaultdict(dict)
    
    for dataset_name, ranking_info in all_rankings.items():
        top_features = set(ranking_info['top_features'])
        data_frame = datasets[dataset_name]
        selected_cols = [col for col in data_frame.columns if col in top_features]

        ds[dataset_name] = data_frame[selected_cols]

    for dataset_name in ['ds7','ds8','ds9','ds10']:
        ds[dataset_name] = datasets[dataset_name]

    return ds
