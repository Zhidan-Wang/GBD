import os
import numpy as np
import random

import torch
import torch.nn as nn

from model import *

import arguments

import utils.load_dataset
import utils.data_loader
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def para(args): 
    if args.dataset == 'yahooR3': 
        args.training_args = {'batch_size': 1024, 'epochs': 500, 'patience': 60, 'block_batch': [6000, 500]}
        #args.base_model_args = {'emb_dim': 10, 'learning_rate': 0.0001, 'weight_decay': 0}
        args.base_model_args = {'emb_dim': 64, 'learning_rate': 0.0001, 'weight_decay': 1, 'disturb_intensity': 1e-3}
        args.dis_model_args = {'emb_dim': 64, 'learning_rate': 1e-1, 'weight_decay': 0}
    elif args.dataset == 'coat':
        args.training_args = {'batch_size': 128, 'epochs': 500, 'patience': 60, 'block_batch': [64, 64]}
        #args.base_model_args = {'emb_dim': 10, 'learning_rate': 0.0001, 'weight_decay': 0.1}
        args.base_model_args = {'emb_dim': 128, 'learning_rate': 0.0001, 'weight_decay': 1, 'disturb_intensity': 1e-4}
        args.dis_model_args = {'emb_dim': 128, 'learning_rate': 1e-1, 'weight_decay': 0}
    elif args.dataset == 'simulation':
        args.training_args = {'batch_size': 1024, 'epochs': 1000, 'patience': 60, 'block_batch': [20, 500]}
        args.base_model_args = {'emb_dim': 128, 'learning_rate': 0.0001, 'weight_decay': 1, 'disturb_intensity': 1e-3}
        args.dis_model_args = {'emb_dim': 128, 'learning_rate': 1e-1, 'weight_decay': 0}
    else: 
        print('invalid arguments')
        os._exit()


def train_and_eval(train_data, val_data, test_data, device = 'cuda',
        base_model_args: dict = {'emb_dim': 64, 'learning_rate': 0.05, 'weight_decay': 0.05},
        dis_model_args: dict = {'emb_dim': 64, 'learning_rate': 0.05, 'weight_decay': 0.05},
        training_args: dict =  {'batch_size': 1024, 'epochs': 100, 'patience': 20, 'block_batch': [1000, 100]}):
    
    # build data_loader. 
    train_loader = utils.data_loader.Block(train_data, u_batch_size=training_args['block_batch'][0], i_batch_size=training_args['block_batch'][1])
    val_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(val_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    test_loader = utils.data_loader.DataLoader(utils.data_loader.Interactions(test_data), batch_size=training_args['batch_size'], shuffle=False, num_workers=0)
    train_data = train_data.cpu()
    train_dense = train_data.to_dense().numpy()
    train_dense = np.where(train_dense != 0, 1, 0)
    #print("sum:", np.sum(train_dense, axis= 0))
    print("user:", train_data.shape[0])
    print("item:", train_data.shape[1])

    #print("train_dense::::", train_dense)

    # data shape
    n_user, n_item = train_data.shape

    # model and its optimizer.
    base_model = MF(n_user, n_item, dim=base_model_args['emb_dim'], dropout=0).to(device)
    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=base_model_args['learning_rate'], weight_decay=0)

    discriminator_s = Discriminator_S(n_user, n_item, dis_model_args['emb_dim']).to(device)
    dis_optimizer_s = torch.optim.SGD(discriminator_s.parameters(), lr=dis_model_args['learning_rate'],
                                    weight_decay=0)
    
    # loss_criterion
    criterion = nn.MSELoss(reduction='sum')
    none_criterion = nn.MSELoss(reduction='mean')
    # begin training
    stopping_args = Stop_args(patience=training_args['patience'], max_epochs=training_args['epochs'])
    early_stopping = EarlyStopping(base_model, **stopping_args)
    for epo in range(early_stopping.max_epochs):
        training_loss = 0
        discriminator_loss = 0
        train_f = 0
        for u_batch_idx, users in enumerate(train_loader.User_loader): 
            for i_batch_idx, items in enumerate(train_loader.Item_loader):
                train_f += 1
                # step 1: Bias-sensitive Forward (calculate discriminator_loss)
                # all pair data in this block

                selec_pred = discriminator_s(base_model.user_latent.weight, base_model.item_latent.weight).to(device).to(torch.float32)
                selec_labels = torch.from_numpy(train_dense).to(device).to(torch.float32)
                dis_loss = none_criterion(selec_pred, selec_labels)
                dis_loss = dis_loss.to(torch.float32)
                dis_loss.backward(retain_graph=True)

                # step2: calculate perturbations
                #print("grad:", base_model.item_latent.weight.grad)
                attack_disturb1 = base_model.item_latent.weight.grad.detach_()
                norm1 = attack_disturb1.norm(dim=-1, p=2)
                norm_disturb1 = attack_disturb1 / (norm1.unsqueeze(dim=-1) + 1e-10)
                disturb1 = base_model_args['disturb_intensity'] * norm_disturb1
                base_model.item_latent.weight.data = base_model.item_latent.weight.data - disturb1

                attack_disturb2 = base_model.user_latent.weight.grad.detach_()
                norm2 = attack_disturb2.norm(dim=-1, p=2)
                norm_disturb2 = attack_disturb2 / (norm2.unsqueeze(dim=-1) + 1e-10)
                disturb2 = base_model_args['disturb_intensity'] * norm_disturb2
                base_model.user_latent.weight.data = base_model.user_latent.weight.data - disturb2

                # step 3: update the disturbed base model
                base_model.train()
                users_train, items_train, y_train = train_loader.get_batch(users, items)
                y_hat = base_model(users_train, items_train)
                loss = criterion(y_hat, y_train) + base_model_args['weight_decay'] * base_model.l2_norm(users_train, items_train)

                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()

                training_loss += loss.item()
                if train_f == 4:
                    train_f = 0
                    #update the discriminator
                    discriminator_s.train()
                    selec_pred = discriminator_s(base_model.user_latent.weight, base_model.item_latent.weight).to(
                        device).to(torch.float32)
                    selec_labels = torch.from_numpy(train_dense).to(device).to(torch.float32)
                    dis_loss1 = none_criterion(selec_pred, selec_labels)
                    dis_loss1 = dis_loss1.to(torch.float32)
                    dis_optimizer_s.zero_grad()
                    dis_loss1.backward()
                    dis_optimizer_s.step()

                    discriminator_loss += dis_loss1.item()

            
        base_model.eval()
        with torch.no_grad():
            # train metrics
            train_pre_ratings = torch.empty(0).to(device)
            train_ratings = torch.empty(0).to(device)
            for u_batch_idx, users in enumerate(train_loader.User_loader): 
                for i_batch_idx, items in enumerate(train_loader.Item_loader): 
                    users_train, items_train, y_train = train_loader.get_batch(users, items)
                    pre_ratings = base_model(users_train, items_train)
                    train_pre_ratings = torch.cat((train_pre_ratings, pre_ratings))
                    train_ratings = torch.cat((train_ratings, y_train))

            # validation metrics
            val_pre_ratings = torch.empty(0).to(device)
            val_ratings = torch.empty(0).to(device)
            for batch_idx, (users, items, ratings) in enumerate(val_loader):
                pre_ratings = base_model(users, items)
                val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
                val_ratings = torch.cat((val_ratings, ratings))
            
        train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['NLL'])
        val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['NLL', 'AUC'])

        print('Epoch: {0:2d} / {1}, Traning: {2}, Validation: {3}'.
                format(epo, training_args['epochs'], ' '.join([key+':'+'%.3f'%train_results[key] for key in train_results]), 
                ' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))

        if early_stopping.check([val_results['AUC']], epo):
            break

    # testing loss
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    base_model.load_state_dict(early_stopping.best_state)

    # validation metrics
    val_pre_ratings = torch.empty(0).to(device)
    val_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(val_loader):
        pre_ratings = base_model(users, items)
        val_pre_ratings = torch.cat((val_pre_ratings, pre_ratings))
        val_ratings = torch.cat((val_ratings, ratings))

    # test metrics
    test_users = torch.empty(0, dtype=torch.int64).to(device)
    test_items = torch.empty(0, dtype=torch.int64).to(device)
    test_pre_ratings = torch.empty(0).to(device)
    test_ratings = torch.empty(0).to(device)
    for batch_idx, (users, items, ratings) in enumerate(test_loader):
        pre_ratings = base_model(users, items)
        test_users = torch.cat((test_users, users))
        test_items = torch.cat((test_items, items))
        test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
        test_ratings = torch.cat((test_ratings, ratings))

    val_results = utils.metrics.evaluate(val_pre_ratings, val_ratings, ['NLL', 'AUC'])
    test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings, ['NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users, items=test_items)
    print('-'*30)
    print('The performance of validation set: {}'.format(' '.join([key+':'+'%.3f'%val_results[key] for key in val_results])))
    print('The performance of testing set: {}'.format(' '.join([key+':'+'%.3f'%test_results[key] for key in test_results])))
    print('-'*30)
    return val_results,test_results

if __name__ == "__main__": 
    args = arguments.parse_args()
    para(args)
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, unif_train, validation, test = utils.load_dataset.load_dataset(data_name=args.dataset, type = 'explicit', seed = args.seed, device=device)
    train_and_eval(train, validation, test, device, base_model_args = args.base_model_args, dis_model_args = args.dis_model_args, training_args = args.training_args)
