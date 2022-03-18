import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# 加载数据、拆分数据集
dataset = TUDataset(root='/home/jerry/TCBB/AutoGNAS/autognas/datasets/MUTAG', name='MUTAG')
dataset = dataset.shuffle()
n = len(dataset) // 10
test_dataset = dataset[:n]
train_dataset = dataset[n:]
test_loader = DataLoader(test_dataset, batch_size=60)
train_loader = DataLoader(train_dataset, batch_size=60)

# 构建模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 128)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        self.pool1 = SAGPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = SAGPooling(128, ratio=0.8)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = SAGPooling(128, ratio=0.8)
        # self.pool3 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))

        # pooling 层: 图数据收缩
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        # readout 层得到图表示
        # 1.gmp: global max pooling 得到图表示,
        # 2.gap: global add pooling 得到图表示.
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # merge 层; 融合各GNN层 readout 结果
        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.lin2(x))

        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

# 训练与评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train(epoch):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)
for epoch in range(1, 201):
    loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))