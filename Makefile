.PHONY: clean-runs
clean-runs:
	@(rm -rf runs)

.PHONY: clean-dataset
clean-dataset:
	@(rm -rf _dataset)

clean: clean-runs clean-dataset

.PHONY: train
train:
	@(python train.py)
