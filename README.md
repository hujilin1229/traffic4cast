# Traffic4cast–Traffic Map Movie Forecasting
by Team MIE-Lab: Henry Martin, Ye Hong, Dominik Bucher, Christian Rupprecht, René Buffat 


## Howto
### Train a model
- Copy the competition raw data into `data_raw` folder such that the top-level city folders are visible. 
- Training configuration can be changed in the `config_unet`file (e.g., choose a city).
- Run the `training_unet.py` file in the `traffic4cast`folder
- The training is logged via tensorboard in the `runs` folder

### Create a submission with the pretrained models
- Download the pretrained models using this [link](https://polybox.ethz.ch/index.php/s/qMMweI8P65HW0h8)
- Store the 3 top-level model folders (e.g., `UNet_Berlin`) in the runs folder (you can overwrite the existing ones)
- Run the `example_create_3modelsubmission.py` script.




