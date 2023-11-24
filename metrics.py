import matplotlib.pyplot as plt

def plot_history(history, label, validation_data=True):
    # list all data in history
    print(history.history)

    accuracy_key = 'accuracy'
    accuracy_val_key = 'val_accuracy'

    if 'accuracy' not in history.history:
        accuracy_key = 'acc'
        accuracy_val_key = 'val_acc'

    # summarize history for accuracy
    plt.plot(history.history[accuracy_key])
    if validation_data:
        plt.plot(history.history[accuracy_val_key])
    plt.title("{} accuracy".format(label))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    if validation_data:
        plt.legend(['train', 'test'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')

    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    if validation_data:
        plt.plot(history.history['val_loss'])
    
    plt.title("{} loss".format(label))
    plt.ylabel('loss')
    plt.xlabel('epoch')

    if validation_data:
        plt.legend(['train', 'test'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')

    plt.show()
