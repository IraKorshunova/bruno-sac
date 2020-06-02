# BrunoSAC


### Requirements

The code was used with the following settings:

- python3
- tensorflow-gpu==1.14.0
- tensorflow-probability==0.7.0
- gym==0.17.1
- mujoco-py== 2.0.2.9
- mujoco200


### Training and testing

To train and then test BrunoSAC on Cheetah-Dir run:

```
python meta-cheetah-dir.py --train 
python meta-cheetah-dir.py --test

```

Similarly, for the oracle:

```
python meta-cheetah-dir.py --train --oracle
python meta-cheetah-dir.py --test --oracle

```

To plot the learning curves and test rewards: 

```
python -m plots.plot_train_cheetah_dir 
python -m plots.plot_test_cheetah_dir

```

The same commands can be used with ```meta_cheetah_vel.py``` for the Cheetah-Vel experiments.


