from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import tensorflow as tf

class CustomCallback(Callback):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get("accuracy")
        self.log[epoch] = logs  
        if accuracy is not None and accuracy >= 0.90:
            print(f"Accuracy reached {accuracy:.2f} at epoch {epoch + 1}.")
            self.model.stop_training = True


def scheduler(epoch, lr):
    if epoch > 0 and epoch % 10 == 0:
        return lr * 0.1
    return lr

lr_scheduler = LearningRateScheduler(scheduler, verbose=1)

