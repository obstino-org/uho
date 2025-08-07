Preparing the dataset is optional. You may clone the processed dataset from https://huggingface.co/datasets/blko/artur1_0

Dataset preprocessing is a method of extracting log-mel spectrograms together with transcriptions.
These are saved as compressed ".parquet" files to save space on disk and enable dataset streaming when training.

Currently the Jupyter notebook assumes that Artur 1.0 dataset is saved in "/mnt/e/datasets/Artur_1_0_full"
You will want to modify that path to directory of your choosing
	--> do that by changing "dataset_path" variable in "get_list_of_transcript_wav_pairs" function

You will also want to modify the following variables:
	-dataset_save_path	(this is where parquet files will be saved)
	-cache_base_path
	-train_cache_path
	-valid_cache_path
	-test_cache_path

Choose any directory that you find appropriate.
