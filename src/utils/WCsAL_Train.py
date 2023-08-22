import os
import numpy as np 
import torch
import torch.optim as optim

from src.utils.GrOWL_utils import sparsity_info, similarity_info, metrics_tr

###################################
####### MULTI-TASK Training #######
###################################

from src.utils.WCsAL_utils import inner_optimization, get_sequence

from src.utils.lsuv import lsuv_init

def train_multitask_model(train_loader, val_loader, model,
                          params_init, init_model = True):
    
    device, w, a, epsilon = params_init["device"], params_init["w"], params_init["a"], params_init["epsilon"]
    num_tasks, num_outs, num_batches = params_init["num_tasks"], params_init["num_outs"], params_init["num_batchEpoch"]
    max_iter_retrain, max_iter_search, num_epochs, tol_epochs = params_init["max_iter_retrain"], params_init["max_iter_search"], params_init["num_epochs"], params_init["tol_epochs"]
    num_model, main_dir, mod_logdir = params_init["num_model"], params_init["main_dir"], params_init["mod_logdir"]
    is_search, min_sparsRate = params_init["is_search"], params_init["min_sparsRate"]
    
    # file names
    if not os.path.exists("%s/%s"%(main_dir, mod_logdir)):
            os.makedirs("%s/%s"%(main_dir, mod_logdir))
    MODEL_FILE = str("%s/%s/model%03d.pth"%(main_dir, mod_logdir, num_model))
    DRAFT_MODEL_FILE = str("%s/%s/draft_model%03d.pth"%(main_dir, mod_logdir, num_model))

    # global variables
    violation_epochs = 0
    
    seq_batch = get_sequence(len(train_loader), num_batches)

    #--------------------------------------------------------------------------#
    # START ALGORITHM                                                          #
    #--------------------------------------------------------------------------#
    
    #### INITIALIZATION
    mu, rho = params_init["mu"], params_init["rho"]
    #### STARTING POINT
    max_iter = params_init["max_iter"]
    if params_init["w"][0] > 0 and params_init["Sparsity_study"]:
        if is_search:
            max_iter = max_iter_search
            # Create the model
            model = model.to(device)
            # Initialize the networks
            if init_model:
                model = lsuv_init(model, train_loader, needed_std=1.0, std_tol=0.1,
                                  max_attempts=10, do_orthonorm=True, device=device) # initialize the model weights
        else:
            max_iter = max_iter_retrain
            zero_layers = params_init["zero_layers"]
            # similarity_m = params_init["similarity_m"]
    else:
        model = model.to(device)

    # Initialize the Langrange nultiplier lambda
    
    lmbd = torch.ones(num_tasks, requires_grad = False)/num_tasks

    #### LOOP
    cont = True
    iterate = 0
    ALL_TRAIN_LOSS = []
    ALL_VAL_ACCU = []
    ALL_ORIG_losses = []
    MODEL_VAL_ACCU = []
    best_avg_val_accu = 0.0
    BEST_val_accu = 0.0
    k = 0
    
    lr, lr_sched_coef = params_init["lr"], params_init["lr_sched_coef"]
    base_optimizer = params_init["base_optimizer"]
    LR_scheduler = params_init["LR_scheduler"]
    criterion = params_init["criterion"]
    optimizer = base_optimizer(model.mt_parameters(), lr=lr)
    if LR_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_sched_coef)
    
    act_bst_accu = 0.0
    best_exist = False
    best_accu_search = 0.0
    succeed_bst = False
    while(k < max_iter):
        print("-------------------------------------")
        print(f"------ Algorithm Iteration {k+1}/{max_iter} ------")
        print("-------------------------------------")
        ### FIRST STEP: find best model weights
        TRAIN_LOSS = []
        VAL_ACCU = []
        ORIG_losses = []
        epfinal = k*(num_epochs*num_batches + num_batches)
        for i in range(num_epochs):
            print("######################")
            print(f"#### EPOCH No {i+1}/{num_epochs} ####")
            print("######################")
            start_batch = epfinal + i*num_batches
            list_batch = seq_batch(start_batch)
            # epsilon = optimizer.param_groups[0]['lr']
            act_state = [k, i]
            orig_train_losses, train_loss, val_accuracy, contrs_wc_after = inner_optimization(model, params_init, optimizer, list_batch, w, a, epsilon, criterion, train_loader, val_loader, device, num_tasks,
                                    mu, lmbd,  act_bst_accu, best_exist)
            
            ORIG_losses.append(orig_train_losses.numpy())
            TRAIN_LOSS.append(train_loss)
            VAL_ACCU.append(val_accuracy.numpy())
            MODEL_VAL_ACCU.append(val_accuracy.mean().item())
            
            if params_init["w"][0] > 0 and params_init["Sparsity_study"]:
                if is_search:
                    sparsity_RT, _ = sparsity_info(model, verbose = False)
                    if (sparsity_RT >= min_sparsRate) and (best_accu_search < val_accuracy.mean().item()):
                            succeed_bst = True
                            best_accu_search = val_accuracy.mean().item()
                            torch.save(model.state_dict(), MODEL_FILE)

            if(best_avg_val_accu < val_accuracy.mean().item()):
                if not is_search:
                    succeed_bst = True
                    torch.save(model.state_dict(), MODEL_FILE)
                else:
                    torch.save(model.state_dict(), DRAFT_MODEL_FILE)
                best_avg_val_accu = val_accuracy.mean().item()
                BEST_val_accu = val_accuracy.numpy()
                BEST_contrs_after_optim = contrs_wc_after
                Best_iter = [k, i]
                act_bst_accu = best_avg_val_accu
                best_exist = True
                violation_epochs = 0
                print("Best global performance (Accuracy)!")
                if (num_tasks-1 <= 3):# print out per-task accuracies for problem with small number of main tasks
                    for i in range(num_tasks-1):
                        print("Accuracy Task {}: {:.04f}%".format(i+1, val_accuracy[i].item()))
            else:
                violation_epochs = violation_epochs + 1
                if tol_epochs is not None:
                    if (violation_epochs > tol_epochs):
                        print(f"No improvement in accuracy after {tol_epochs} more epochs. END!!!")  
                        break;
        
        print("Learning rate used: ", optimizer.param_groups[0]['lr'])
        #--------------------------------------------------------------------------#
        # update learning rate scheduler                                           #
        #--------------------------------------------------------------------------#
        if LR_scheduler:
            lr_scheduler.step()

        ALL_TRAIN_LOSS.append(TRAIN_LOSS)
        ALL_VAL_ACCU.append(VAL_ACCU)
        ALL_ORIG_losses.append(ORIG_losses)
        ### SECOND STEP
        lmbd = lmbd.reshape(-1, 1).to(device) + mu*BEST_contrs_after_optim.reshape(-1, 1).to(device)
            
        print("Penalty coefficient (mu) used: ", mu)
        ### THIRD STEP: Update lambda
        mu = rho*mu
        
        ### STOP after a given "maximum number of iterations" 
        k = k + 1
   
    #### RETURN FINAL (BEST) MODEL
    if succeed_bst:
        model.load_state_dict(torch.load(MODEL_FILE))
    else:
        print("Could not find a model with the required sparsity rate!\n The model with the highest accuracy has been returned!")
        model.load_state_dict(torch.load(DRAFT_MODEL_FILE))
        
    tr_metrics, _ = metrics_tr(model, verbose = False)
            
    return model, [succeed_bst, tr_metrics], ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter


def full_training(train_loader, val_loader, model,
                          params_init, init_model = True):
    if params_init["Sparsity_study"]:
        print("################################")
        print(f"#### SPARSITY inducing ... ####")
        print("################################")
    else:
        print("################################")
        print(f"#### TRAINING started ! ####")
        print("################################")
    params_init["is_search"] = True
    model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter = train_multitask_model(train_loader, val_loader, model,
                          params_init, init_model = init_model)
    
    if params_init["w"][0] > 0 and params_init["Sparsity_study"]:
        _, ZERO_layers = sparsity_info(model)
        print("Computing similarity matrices . . . ")
        similarity_M = similarity_info(model, zero_layers = ZERO_layers)
        print("Done !")

        params_init["zero_layers"] = ZERO_layers
        params_init["similarity_m"] = similarity_M
    
        print("###############################")
        print(f"#### RETRAINING started ! ####")
        print("###############################")
        params_init["is_search"] = False    
        model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter =  train_multitask_model(train_loader, val_loader, model,
                              params_init, init_model = False)
        
    return model, TR_metrics, ALL_TRAIN_LOSS, ALL_VAL_ACCU, ALL_ORIG_losses, MODEL_VAL_ACCU, BEST_val_accu, Best_iter



###################################
####### SINGLE-TASK Training ######
###################################


def train_single_model(train_loader, val_loader, model, params_sg):
    
    device, ind_task = params_sg["device"], params_sg["ind_task"]
    num_epochs, tol_epochs = params_sg["num_epochs"], params_sg["tol_epochs"]
    main_dir, mod_logdir = params_sg["main_dir"], params_sg["mod_logdir"]
    num_model = params_sg["num_model"]
    lr_step_size = params_sg["lr_step_size"]
    lr_sched_coef = params_sg["lr_sched_coef"]

    # number of epochs
    NUM_EPOCHS = num_epochs

    # file names
    if not os.path.exists("%s/%s"%(main_dir, mod_logdir)):
            os.makedirs("%s/%s"%(main_dir, mod_logdir))
    MODEL_FILE = str("%s/%s/sg_model%03d.pth"%(main_dir, mod_logdir, num_model))

    # global variables
    if tol_epochs is None: tol_epochs = num_epochs
    violation_epochs = 0
    
    lr = params_sg["lr"]
    #mmt = params_sg["momentum"] 
    base_optimizer = params_sg["base_optimizer"]
    LR_scheduler = params_sg["LR_scheduler"]
    criterion = params_sg["criterion"] 
    
    if base_optimizer == optim.SGD:
        print("\nSGD will be used as optimizer.\n")
        momentum = params_sg["momentum"]
        optimizer = base_optimizer(model.parameters(), lr=lr, momentum = momentum)
        if LR_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_sched_coef)
    else:
        optimizer = base_optimizer(model.parameters(), lr=lr)
        if LR_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_sched_coef)

    # global variables
    g_step = 0
    max_correct = 0

    model = model.to(device)

    # training and evaluation loop
    for epoch in range(NUM_EPOCHS):
        print("----------------------------------")
        print(f"-------- EPOCH {epoch+1}/{NUM_EPOCHS} --------")
        print("----------------------------------")
        #--------------------------------------------------------------------------#
        # train process                                                            #
        #--------------------------------------------------------------------------#
        model.train()
        train_loss = 0
        train_corr = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target[ind_task].to(device, dtype=torch.int64)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_pred = output.argmax(dim=1, keepdim=True)
            train_corr += train_pred.eq(target.view_as(train_pred)).sum().item()
            train_loss += criterion(output, target, reduction='sum').item()
            loss.backward()
            optimizer.step()
            g_step += 1

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss /= len(train_loader.dataset)
        # train_accuracy = 100 * train_corr / len(train_loader.dataset)
            
        #--------------------------------------------------------------------------#
        # test process                                                             #
        #--------------------------------------------------------------------------#
        model.eval()
        val_loss = 0
        correct = 0
        total_pred = np.zeros(0)
        total_target = np.zeros(0)
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target[ind_task].to(device,  dtype=torch.int64)
                output = model(data)
                val_loss += criterion(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                total_pred = np.append(total_pred, pred.cpu().numpy())
                total_target = np.append(total_target, target.cpu().numpy())
                correct += pred.eq(target.view_as(pred)).sum().item()
            if(max_correct < correct):
                torch.save(model.state_dict(), MODEL_FILE)
                max_correct = correct
                print("Best accuracy! correct images: %5d"%correct)

        #--------------------------------------------------------------------------#
        # output                                                                   #
        #--------------------------------------------------------------------------#
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / len(val_loader.dataset)
        best_val_accuracy = 100 * max_correct / len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) (best: {:.2f}%)\n'.format(val_loss, correct, len(val_loader.dataset), val_accuracy, best_val_accuracy))

        #--------------------------------------------------------------------------#
        # update learning rate scheduler                                           #
        #--------------------------------------------------------------------------#
        if LR_scheduler:
            if epoch % lr_step_size == 0:
                lr_scheduler.step()
        
    model.load_state_dict(torch.load(MODEL_FILE))
        
    return model, train_loss, val_accuracy
