# Running Instructions
- In order to train the model on blue crystal 4, run the `train_saliency.sh` script. This will return `preds.pkl`, and a checkpoint file `checkpoints.pt`.
- To get the accuracy results, run `eval.sh` on blue crystal 4, and look at the corresponding slurm file for results.
- To get all output images, with ground truth and slaiency predictions, run `visual.sh` on blue crystal 4. All 500 validation image will be displayed along with ground truth and prediction in the `images` folder.
- To re-produce our filter images, run `filter_vis.py`, which will load `checkpoints.pt` and output `filters.png`.
