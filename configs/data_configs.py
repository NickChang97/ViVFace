from configs import transforms_config
from configs.paths_config import dataset_paths
DATASETS = {
	
	'aligned_celevb_image': {
		"root_dir": "/xxx/xxx", # specify training dir
		'transforms': None,
		"frame_shape": (256, 256, 3), 
		"is_train": True,
        "random_seed": 0,
        "sample_percent":1.0
	},
	'specify_source_image_driving_dir': {
		"source_image": 'xxx.png',
		"driving_dir": '/xxx/xxx',
	},
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'cars_encode': {
		'transforms': transforms_config.CarsEncodeTransforms,
		'train_source_root': dataset_paths['cars_train'],
		'train_target_root': dataset_paths['cars_train'],
		'test_source_root': dataset_paths['cars_test'],
		'test_target_root': dataset_paths['cars_test'],
	},
	'horse_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['horse_train'],
		'train_target_root': dataset_paths['horse_train'],
		'test_source_root': dataset_paths['horse_test'],
		'test_target_root': dataset_paths['horse_test'],
	},
	'church_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['church_train'],
		'train_target_root': dataset_paths['church_train'],
		'test_source_root': dataset_paths['church_test'],
		'test_target_root': dataset_paths['church_test'],
	},
	'cats_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['cats_train'],
		'train_target_root': dataset_paths['cats_train'],
		'test_source_root': dataset_paths['cats_test'],
		'test_target_root': dataset_paths['cats_test'],
	}
}
