def print_metrics(train_split_percentage, history, optimizer,learning_rate, steps_per_epoch, total_epochs, validation_steps, batch_size, print_table_header = False):
  """Prints the model metrics in a Google Colab Notebook documentable format
  Args:
    train_split_percentage: `int` for the percentage of the dataset allocated to print
    model: `Model` object containing the trained model
    history: `History` object returned by the fit function after training completion
    optimzer: `optimizer` object for tensorflow used for optimization during training
    learning_rate: `float` for the learning rate
    steps_per_epoch: `int` for batches tried per one epoch
    total_epochs: `int` for total number of epochs
    batch_size: `int` for the images per batch
    print_table_header: `bool` indicating whether to print values along with the table header
  Returns:
    None
  """
  acc = history.history['acc']
  val_acc = history.history['val_acc']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(acc))

  optimizer_name = 'SGD'
  print("copy the following text to your notebook text block: ")
  if(print_table_header):
    print(">Split Size | Batch Size | Epochs | Steps Per Epoch | Validation Steps | Optimizer | Learning Rate")
    print(">--- | --- | --- | --- | --- | --- | ---")
  print(">%02d-%02d | %02d | %02d | %02d | %02d | %s | %f " %
        (int(train_split_percentage), (100 - int(train_split_percentage)),
         batch_size,
         total_epochs,
         steps_per_epoch,
         validation_steps,
         optimizer_name,
         learning_rate))


  print(">$Loss_{train} =  %.4f$ | | $Loss_{val} = %.4f$ |  |$Accuracy_{train} = %.4f$ | |  $Accuracy_{val} = %.4f$" %
        (loss[-1],
         val_loss[-1],
         acc[-1],
         val_acc[-1]))
