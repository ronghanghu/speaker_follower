convolutional_feature_store_paths = {
    'imagenet': 'img_features/imagenet_convolutional',
    'places365': 'img_features/places365_convolutional',
}

mean_pooled_feature_store_paths = {
    'imagenet': 'img_features/ResNet-152-imagenet.tsv',
    'places365': 'img_features/ResNet-152-places365.tsv',
}

bottom_up_feature_store_path = "img_features/bottom_up_10_100"
bottom_up_feature_cache_path = "img_features/bottom_up_10_100.pkl"
bottom_up_feature_cache_dir = "img_features/bottom_up_10_100_cache"

bottom_up_attribute_path = "data/visual_genome/attributes_vocab.txt"
bottom_up_object_path = "data/visual_genome/objects_vocab.txt"
