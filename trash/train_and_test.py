from torch.autograd import Variable

def train(epoch):

  for batch_idx, (image, label) in enumerate(train_loader):
    image, label = Variable(image), Variable(label)
    optimizer.zero_grad()
    output = net(image, label)
    print(output.size())
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    print('epoch: {}\t Loss: {}'.format(epoch , loss.data))
    #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(image), len(train_loader.dataset),
    #                                                       100 * batch_idx / len(train_loader), loss.data[0]))

def test():

  for (image, label) in test_loader:
    image, label = Variable(image.float(), volatile=True), Variable(label)
    output = model(image)
    test_loss += criterion(output, label).data[0]
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()

  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

