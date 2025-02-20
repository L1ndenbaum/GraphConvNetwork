import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATv2Conv
import torch, nets, utils
import matplotlib.pyplot as plt

def train(ensemble_net:nets.EnsembleNet, train_loader, num_epochs, lr, args:utils.Args, 
          weight_decay=0., scheduler_stepsz=10, scheduler_gamma=0.95):

    def init_weights(module): 
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, GCNConv):
            nn.init.xavier_uniform_(module.lin.weight)
        elif isinstance(module, GATv2Conv):
            nn.init.xavier_uniform_(module.lin_l.weight)
            nn.init.xavier_uniform_(module.lin_r.weight)
            nn.init.xavier_uniform_(module.att)

    def get_query_acc(y_hat, y, query_size):
        return torch.sum(y_hat[-query_size*args.N:].argmax(dim=1) == y[-query_size*args.N:]).cpu().numpy() / (query_size*args.N)
    
    ensemble_net.to(args.device)
    ensemble_net.apply(init_weights)
    ensemble_net.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_stepsz, gamma=scheduler_gamma)
    losses, query_accs= [], []

    for epoch in range(1, num_epochs + 1):
        metric = utils.Accumulator(3)  # 单样本损失平均值, 一个Epoch的Query正确率, 一个Epoch批量数
        for data in train_loader:
            optimizer.zero_grad()
            features = data[:-1]
            features = [feature.to(args.device) for feature in features]
            labels = data[-1].squeeze(0).to(args.device)

            if args.in_data_type == 'audio':
                waveforms, spectrograms = features
                waveforms = waveforms.squeeze(0)
                spectrograms = spectrograms.squeeze(0)
                y_hat = ensemble_net(waveforms, spectrograms, labels=labels)
            elif args.in_data_type == 'seq':
                waveforms = features[0]
                waveforms = waveforms.squeeze(0)
                y_hat = ensemble_net(waveforms, labels=labels)

            loss = loss_function(y_hat, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            metric.add(loss.item() / len(labels), get_query_acc(y_hat, labels, args.query_size), 1)
            del features, labels
            torch.cuda.empty_cache()

        print(f"Epoch: {epoch:2d} | Loss: {metric[0]:.7f} | Query Acc: {metric[1] / metric[2] * 100:.2f}%")
        losses.append(metric[0])
        query_accs.append(metric[1] / metric[2])

    _, axes = plt.subplots(1, 2, figsize=(16, 8))
    epoch_list = list(range(1, num_epochs+1))
    axes[0].plot(epoch_list, losses, label='Loss')
    axes[0].set_title('Loss In Epochs')
    axes[0].legend()
    axes[1].plot(epoch_list, query_accs, label='Query Acc')
    axes[1].set_title('Accuracy In Epochs')
    axes[1].legend()

def test(ensemble_net, test_loader, args:utils.Args):

    def get_query_acc(y_hat, y, query_size):
            return torch.sum(y_hat[-query_size*args.N:].argmax(dim=1) == y[-query_size*args.N:]).cpu().numpy() / (query_size*args.N)
    
    ensemble_net.to(args.device)
    ensemble_net.eval()
    query_accs = []

    with torch.no_grad():
        for test_idx, data in enumerate(test_loader):
            features = data[:-1]
            features = [feature.to(args.device) for feature in features]
            labels = data[-1].squeeze(0).to(args.device)

            if args.in_data_type == 'audio':
                waveforms, spectrograms = features
                waveforms = waveforms.squeeze(0)
                spectrograms = spectrograms.squeeze(0)
                y_hat = ensemble_net(waveforms, spectrograms, labels=labels)
            elif args.in_data_type == 'seq':
                waveforms = features[0]
                waveforms = waveforms.squeeze(0)
                y_hat = ensemble_net(waveforms, labels=labels)

            query_accs.append(get_query_acc(y_hat, labels, args.query_size))
            del features, labels
            torch.cuda.empty_cache()
            print(f"Test Batch {test_idx+1:3d}   |   Query Acc:{query_accs[-1]*100:.2f}%")

    test_idx_list = list(range(1, len(test_loader)+1))
    _ = plt.figure(figsize=(16, 8))
    plt.plot(test_idx_list, query_accs, label='Query Acc')
    plt.title('Accuracy In Test Batches')
    plt.legend()
    avg_test_acc = sum(query_accs) / len(query_accs)
    print(f"Average Query Acc:{avg_test_acc*100:.2f}%")

    return avg_test_acc