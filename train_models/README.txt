To train the models, first build a singularity ".sif" image container by running the following:
	sudo singularity build whisper.sif whisper.def

Then in same folder that you have your ".sif" image, create a folder called "./data".
Inside "./data" should be "train.py".
Inside "./data" create a "dataset" folder.
In your shell navigate to dataset folder and run the following:
	git clone https://huggingface.co/datasets/blko/artur1_0
As a result you should have "artur1_0" folder inside your "dataset" folder.
Then navigate to "./" and run train.sh
