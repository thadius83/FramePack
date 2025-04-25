# Extended bucket options with multiple resolutions for low memory usage
bucket_options = {
    # Original high-res buckets
    640: [
        (416, 960),
        (448, 864),
        (480, 832),
        (512, 768),
        (544, 704),
        (576, 672),
        (608, 640),
        (640, 608),
        (672, 576),
        (704, 544),
        (768, 512),
        (832, 480),
        (864, 448),
        (960, 416),
    ],
    
    # Medium resolution buckets (512)
    512: [
        (336, 768),
        (384, 640),
        (416, 608),
        (448, 576),
        (480, 544),
        (512, 512),
        (544, 480),
        (576, 448),
        (608, 416),
        (640, 384),
        (768, 336),
    ],
    
    # Low resolution buckets (384)
    384: [
        (256, 512),
        (288, 480),
        (320, 448),
        (352, 416),
        (384, 384),
        (416, 352),
        (448, 320),
        (480, 288),
        (512, 256),
    ],
    
    # Very low resolution buckets (320)
    320: [
        (192, 384),
        (224, 352),
        (256, 320),
        (288, 288),
        (320, 256),
        (352, 224),
        (384, 192),
    ],
    
    # Extremely low resolution buckets (256)
    256: [
        (160, 320),
        (192, 256),
        (224, 224),
        (256, 192),
        (320, 160),
    ],
}


def find_nearest_bucket(h, w, resolution=640):
    """Find the nearest bucket for the given height and width.
    
    If the specified resolution doesn't exist in bucket_options, 
    it will find the nearest available resolution.
    """
    # If the requested resolution isn't available, find the closest one
    if resolution not in bucket_options:
        available_resolutions = list(bucket_options.keys())
        resolution = min(available_resolutions, key=lambda x: abs(x - resolution))
        print(f"Resolution {resolution} not found, using closest available: {resolution}")
    
    min_metric = float('inf')
    best_bucket = None
    
    for (bucket_h, bucket_w) in bucket_options[resolution]:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    
    return best_bucket


def get_available_resolutions():
    """Returns a list of available resolutions for the UI"""
    return sorted(list(bucket_options.keys()))