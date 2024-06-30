FEATURE_CONFIG = {
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

def get_feature_config():
    return FEATURE_CONFIG