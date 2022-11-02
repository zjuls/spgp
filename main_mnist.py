from model import *
from stca_loss import *
import torchvision
import torchvision.transforms as transforms
import os
import time
import tqdm
import numpy as np
data_path = r'/home/liam' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
num_epochs = 100
batch_size = 150
learning_rate_2 = 1e-3
num_classes = 10
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=False, transform=transforms.Compose([transforms.RandomCrop(28, padding=2), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])) #transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip(),
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=False,  transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])) #transforms.Resize(40),transforms.RandomResizedCrop(size=32),
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

def spgp_mnist(name = 'mnist_spgp'):
    best_acc = 0  # best test accuracy
    acc_record = list([])
    acc_record2 = list([])
    evaluator = nn.CrossEntropyLoss()#STCA_ClassifyLoss()
    snn = MNIST_Net(num_classes, 1, False, 64, spgp=True)
    snn = nn.DataParallel(snn)
    snn.to(device)
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate_2, betas=(0.9, 0.999))

    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        correct2 = 0
        total2 = 0
        for i, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
            snn.zero_grad()
            snn.train()

            images = images.float().to(device)
            # labels_ = torch.zeros(images.size()[0], num_classes).scatter_(1, labels.view(-1, 1), 1)

            labels_ = labels.to(device)

            output = snn(images)
            loss = evaluator(output, labels_)
            optimizer.zero_grad()
            loss.backward()
            running_loss = running_loss + loss.detach().item()
            optimizer.step()

            #predicted2 = output.cpu().sum(-1).argmax(dim=1, keepdim=True)
            predicted2 = output.cpu().argmax(dim=1, keepdim=True)
            total2 += float(labels.size(0))
            correct2 += predicted2.eq(labels.cpu().view_as(predicted2)).sum().item()
            if (i + 1) % 50 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Acc: %.5f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // images.size()[0], running_loss,
                         100. * float(correct2) / float(total2)))
                running_loss = 0

        acc2 = 100. * float(correct2) / float(total2)
        acc_record2.append(acc2)
        print('Iters:', epoch, '\n\n\n')
        print('Test Accuracy of the model on the train set: %.3f' % (100 * correct2 / total2))
        optimizer = lr_scheduler(optimizer, epoch, learning_rate_2, 50)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(test_loader)):
                inputs = inputs.float().to(device)
                # targets_ = torch.zeros(inputs.size()[0], num_classes).scatter_(1, targets.view(-1, 1), 1)
                # targets_ = targets_.to(device)

                optimizer.zero_grad()
                snn.eval()
                output = snn(inputs)
                pred = output.cpu().argmax(dim=1, keepdim=True)
                total += float(targets.size(0))
                correct += pred.eq(targets.cpu().view_as(pred)).sum().item()
                if batch_idx % 50 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
                    #print(output.cpu())
        print('Iters:', epoch, '\n\n\n')
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        if (acc > best_acc):
            best_acc = acc
            best_net = snn.module.state_dict()
            best_epoch = epoch
            print('change best..' + str(best_acc))
            print('Saving..')
            state = {
                'best_net': best_net,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'acc_record': acc_record,
            }
            if not os.path.exists('checkpoint'):
                os.makedirs('checkpoint')
            torch.save(state, './checkpoint/ckpt' + name + '.t7')
        print("best acc is %.3f " % best_acc)
    np.savetxt(name + '_test_acc.txt', acc_record)
    np.savetxt(name + '_train_acc.txt', acc_record2)

if __name__ == '__main__':
    spgp_mnist()
