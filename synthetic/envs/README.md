Usage

First, start visdom in another terminal:
```
visdom
```
After that:
```
cd synthetic/
python train.py --env-name cartpole1 --method crl-envelope --model linear --gamma  0.99 --mem-size 4000 --batch-size 256 --lr  1e-3 --epsilon 0.5 --epsilon-decay --weight-num 32 --episode-num 5000 --optimizer Adam --save crl/envelope/saved/ --log crl/envelope/logs/ --update-freq 100 --beta 0.01 --name cartpole1_0
```