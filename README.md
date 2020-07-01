# BrunoSAC


**Exchangeable Models in Meta Reinforcement Learning**<br>
I. Korshunova, J. Degrave, J. Dambre, A. Gretton, F. Huszár<br>
_Lifelong Learning Workshop at ICML 2020_


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
python meta_cheetah_dir.py --train 
python meta_cheetah_dir.py --test

```

Similarly, for the oracle:

```
python meta_cheetah_dir.py --train --oracle
python meta_cheetah_dir.py --test --oracle

```

To plot the learning curves and test rewards: 

```
python -m plots.plot_train_cheetah_dir 
python -m plots.plot_test_cheetah_dir

```

The same commands can be used with ```meta_cheetah_vel.py``` for the Cheetah-Vel experiments.

### Questions?

Please send an email to `irene.korshunova@gmail.com`, and I'll be happy to answer.