import os
from typing import List


class trainConfig:
    def __init__(
            self,
            lr: List[float],
            batch_size: int,
            epoch: int,
            print_loss: bool,
            pretrain: bool,
            data_dir: str,
            weights_dir: str
    ) -> None:
        self.learning_rate = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.print_loss = print_loss
        self.pretrain = pretrain
        self.data_dir = data_dir
        self.weights_dir = weights_dir

        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)


    #learning_rate = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    #print_loss = True
    #batch_size = 3
    #epoch = 50
    #pretrain = True
    #data_dir = './data'
    #checkpoints = './weights/saved_checkpoints'
    #if not os.path.exists(checkpoints):
    #    os.makedirs(checkpoints)
    #save_best = './weights/best_weight'
    #if not os.path.exists(save_best):
    #    os.makedirs(save_best)
