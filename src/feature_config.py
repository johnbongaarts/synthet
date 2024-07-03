GENRE_FEATURE_CONFIG = {
    'TOTAL_FEATURES': 28,
    'TEMPO': 0,
    'SPECTRAL_CENTROID': 1,
    'SPECTRAL_BANDWIDTH': 2,
    'SPECTRAL_CONTRAST_START': 3,
    'SPECTRAL_CONTRAST_END': 9,
    'SPECTRAL_ROLLOFF': 9,
    'CHROMA_START': 10,
    'CHROMA_END': 15,
    'MFCC_START': 15,
    'MFCC_END': 27,
    'RMS': 27
}

MOOD_FEATURE_CONFIG = {
    'TOTAL_FEATURES': 2,
    'AROUSAL': 0,
    'VALENCE': 1,
}

def get_genre_feature_config():
    return GENRE_FEATURE_CONFIG

def get_mood_feature_config():
    return MOOD_FEATURE_CONFIG

def get_feature_config():
    combined_config = {}
    
    # Start with genre features
    combined_config.update(GENRE_FEATURE_CONFIG)
    
    # Add mood features, adjusting their indices
    mood_offset = GENRE_FEATURE_CONFIG['TOTAL_FEATURES']
    for key, value in MOOD_FEATURE_CONFIG.items():
        if key == 'TOTAL_FEATURES':
            combined_config[key] = GENRE_FEATURE_CONFIG['TOTAL_FEATURES'] + MOOD_FEATURE_CONFIG['TOTAL_FEATURES']
        else:
            combined_config[f'MOOD_{key}'] = value + mood_offset
    
    return combined_config

def print_config(config_name, config_dict):
    print(f"\n{config_name}:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")