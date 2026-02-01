import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.utils import class_weight
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from KANConv import KANConv1DLayer as KANConv
from mamba2 import NdMamba2_1d
import time

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class Classifier(nn.Module):
    def __init__(self, args, output_directory, nb_classes, x_tarin1, x_tarin2, net_emd_tf_s, net_emd_tf_t,
                 net_emd_target_s, net_emd_target_t, device='cuda', verbose=False, build=True, load_weights=False,
                 patience=5):
        super(Classifier, self).__init__()

        self.device = device  # Store the device information
        self.to(device)  # Move model to device (GPU or CPU)


        self.patience = 5
        self.output_directory = output_directory

        self.kan1 = KAN([24, 24])
        self.kan2 = KAN([24, 24])

        self.mamba = NdMamba2_1d(2, 128, 128, True)


        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(24, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)

        # Linear layer to match Transformer input dimension
        self.fc_transformer_input = nn.Linear(640, 768)

        # self.conv = nn.Conv1d(640, 16, 15, padding=7)
        self.conv = KANConv(640, 16, 15, padding=10)

        self.bn = nn.BatchNorm1d(16, momentum=0.8)
        self.pool = nn.MaxPool1d(5, padding=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc_pred2 = nn.Linear(16, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc_final = nn.Linear(128, nb_classes)

    def forward(self, x_train1, x_train2, net_emd_tf_s, net_emd_tf_t, net_emd_target_s, net_emd_target_t):

        # Ensure inputs are on the correct device
        x_train1, x_train2 = x_train1.to(self.device), x_train2.to(self.device)
        net_emd_tf_s, net_emd_tf_t = net_emd_tf_s.to(self.device), net_emd_tf_t.to(self.device)
        net_emd_target_s, net_emd_target_t = net_emd_target_s.to(self.device), net_emd_target_t.to(self.device)

        f1 = x_train1.squeeze(2)
        f2 = x_train2.squeeze(2)
        f1 = f1.unsqueeze(1)
        f2 = f2.unsqueeze(1)


        prot_features = self.kan1(f1)
        drug_features = self.kan2(f2)
        combined_features = torch.cat([prot_features, drug_features], dim=1)
        combined_features = self.mamba(combined_features)


        combined_features = F.relu(self.fc1(combined_features))
        combined_features = self.dropout(combined_features)
        combined_features = F.relu(self.fc2(combined_features))
        input_layer_net_tf_s_ = net_emd_tf_s.squeeze(1)
        input_layer_net_tf_t_ = net_emd_tf_t.squeeze(1)
        input_layer_net_target_s_ = net_emd_target_s.squeeze(1)
        input_layer_net_target_t_ = net_emd_target_t.squeeze(1)
        combined_features = combined_features.mean(dim=1)

        # Concatenate features
        all_features = torch.cat(
            [combined_features, input_layer_net_tf_s_, input_layer_net_tf_t_, input_layer_net_target_s_,
             input_layer_net_target_t_], dim=1)

        # Reshape
        all_features = all_features.unsqueeze(1).permute(0, 2, 1)

        # Convolution and Batch Normalization
        x = self.conv(all_features)
        x = self.bn(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for dense layer

        # Fully connected layers
        x = F.relu(self.fc_pred2(x))
        x = self.dropout(x)
        x = self.fc_final(x)
        return x

    def fit_5CV(self, x_train_1, x_train_2, net_emd_tf_s_train, net_emd_tf_t_train, net_emd_target_s_train,
                net_emd_target_t_train, y_train,
                x_val_1, x_val_2, net_emd_tf_s_val, net_emd_tf_t_val, net_emd_target_s_val, net_emd_target_t_val, y_val,
                x_test_1, x_test_2, net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test, net_emd_target_t_test):

        start_time = time.time()

        batch_size = 128
        nb_epochs = 150

        y_train_num = []
        for i in range(y_train.shape[0]):
            a = y_train[i][0]
            b = y_train[i][1]
            c = y_train[i][2]

            if a == 1:
                y_train_num.append(0)
            elif b == 1:
                y_train_num.append(1)
            elif c == 1:
                y_train_num.append(2)
            else:
                print('error y-train')
        y_train_num = np.array(y_train_num)


        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_num),
            y=y_train_num
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        class_weights = class_weights / class_weights.sum() * len(class_weights)


        print(class_weights)
        print('------------------------------------------------------------------------------')
        mini_batch_size = int(min(x_train_1.shape[0] / 10, batch_size))


        train_dataset = TensorDataset(x_train_1, x_train_2, net_emd_tf_s_train, net_emd_tf_t_train,
                                      net_emd_target_s_train, net_emd_target_t_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)

        val_dataset = TensorDataset(x_val_1, x_val_2, net_emd_tf_s_val, net_emd_tf_t_val, net_emd_target_s_val,
                                    net_emd_target_t_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=False)

        optimizer = optim.AdamW(self.parameters(),
                              lr=1e-3,
                              weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_acc = 0

        for epoch in range(nb_epochs):

            epoch_start_time = time.time()

            self.train()
            total_loss = 0
            for inputs in train_loader:
                # Unpack your inputs depending on your specific needs
                x1, x2, emd_s, emd_t, target_s, target_t, labels = inputs
                x1, x2 = x1.to(self.device), x2.to(self.device)
                emd_s, emd_t = emd_s.to(self.device), emd_t.to(self.device)
                target_s, target_t = target_s.to(self.device), target_t.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(x1.permute(0, 2, 1), x2.permute(0, 2, 1), emd_s, emd_t, target_s, target_t)
                _, labels = torch.max(labels, 1)
                loss = criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)  # 添加这行

                optimizer.step()
                total_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch}, Batch: {i}, Current LR: {current_lr}')

            self.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for inputs in val_loader:
                    x1, x2, emd_s, emd_t, target_s, target_t, labels = inputs
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    emd_s, emd_t = emd_s.to(self.device), emd_t.to(self.device)
                    target_s, target_t = target_s.to(self.device), target_t.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self(x1.permute(0, 2, 1), x2.permute(0, 2, 1), emd_s, emd_t, target_s, target_t)
                    _, labels = torch.max(labels, 1)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.state_dict(), self.output_directory + 'best_model.pth')

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)},Val Accuracy: {val_acc}, {epoch_duration:.2f} seconds.')



        y_pred = self.predict(
            x_test_1, x_test_2, net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test, net_emd_target_t_test)
        yy_pred = np.argmax(y_pred, axis=1)

        end_time = time.time()
        total_duration = end_time - start_time
        print(f'Total training time: {total_duration:.2f} seconds.')

        return y_pred, yy_pred

    def predict(self, x_test_1, x_test_2, net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,
                net_emd_target_t_test):
        test_dataset = TensorDataset(x_test_1, x_test_2, net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,
                                     net_emd_target_t_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # Load the best model
        self.load_state_dict(torch.load(f'{self.output_directory}/best_model.pth'))
        # Test the model
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                x1, x2, emd_s, emd_t, target_s, target_t = inputs
                predicted = self(x1.permute(0, 2, 1), x2.permute(0, 2, 1), emd_s, emd_t, target_s, target_t)
                predictions.extend(predicted.cpu().numpy())
        return predictions