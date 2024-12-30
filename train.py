import time

def train(num_epochs, train_loader, val_loader, device):
    train_losses = []
    val_losses = []
    total_time = 0
    for epoch in range(num_epochs):
        tic = time.time()
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total += loss.item()

        train_loss = total / len(train_loader)
        train_losses.append(train_loss)

        # Calcuating the loss for the validation set
        total = 0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total += loss.item()

        val_loss = total / len(val_loader)
        val_losses.append(val_loss)

        toc = time.time()
        total_time += toc - tic
        print('Epoch:', epoch + 1, '/', num_epochs, '\tDuration:', round((toc - tic), 2), 'secs', '\tTrain Loss:', round(train_loss, 4), '\tVal Loss:', round(val_loss, 4))

        toc = time.time()

    print('\nTotal Running time:', round((total_time) / 60, 2), 'mins')
    return train_losses, val_losses
