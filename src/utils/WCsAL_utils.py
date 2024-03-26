import torch

from src.utils.GrOWL_utils import apply_growl, force_sharing
from src.utils.GrOWL_utils import  keep_sparse

#########################################################################################
#####  Helper functions for our method combining WC scalarization and AL method #####
#########################################################################################

# Define the Weighted Chebyshev Scalarization contraints
def contraints(model, w, a, epsilon, device, outputs, targets, criterion, sparsity_objective):
    
    num_tasks = len(outputs) + 1

    # Compute the individual losses (Objectives)
    # Compute the WCS contraints
    indiv_orig_LOSSES = []
    CONTRS_1 = []
    sum_contr_1 = 0.0
    for i in range(num_tasks):
        if i == 0:
            loss_i = sparsity_objective
        else:
            loss_i =  criterion(outputs[i-1], targets[:, i-1]).to(device)
        indiv_orig_LOSSES.append(loss_i.detach().cpu().item())
        contr_i_1 = w[i]*(loss_i - a[i]) - model.wc_variable
        sum_contr_1 = sum_contr_1 + epsilon*(loss_i - a[i])
        CONTRS_1.append(contr_i_1)
    CONTRS_1 = torch.stack(CONTRS_1).reshape(-1,1).to(device)
    
    CONTRS_2 = [w[i]*sum_contr_1 for i in range(num_tasks)]
    CONTRS_2 = torch.stack(CONTRS_2).reshape(-1,1).to(device)
    
    CONTRS = CONTRS_1 + CONTRS_2
    return CONTRS, torch.tensor(indiv_orig_LOSSES).reshape(-1,1)

# Define the Augmented Lagrangian criterion 
def Augmented_Lagrangian_criterion(model, device, CONTRS, mu, lmbd):
    
    lmbd = lmbd.reshape(-1, 1).to(device)
    CONTRS = CONTRS.reshape(-1, 1).to(device)
    
    # Compute the Augmented Lagrangian Function
    scalar_product = torch.matmul(lmbd.T, CONTRS)
    penalty_func = (mu/2)*torch.norm(CONTRS, p=2)**2
    Augmented_lagrangian = model.wc_variable + scalar_product.to(device) + penalty_func.to(device)

    return Augmented_lagrangian.to(device)

def get_sequence(N, p):
    def sequence(n):
        if (n < 0):
            raise ValueError("n must be a non-negative integer.")
        if n + p <= N:
            return list(range(n, n+p))
        elif n < N:
            return list(range(n, N+1)) + list(range(1, p-N+n))
        elif n >= N:
            return sequence(n-N)  
    return sequence

# def inner_optimization(model, params_init, optimizer, list_batch, w, a, epsilon, criterion, train_loader, val_loader, device, num_tasks, mu, lmbd, act_bst_accu, best_exist = False):
def inner_optimization(model, params_init, optimizer, w, a, epsilon, criterion, train_loader, val_loader, device, num_tasks, mu, lmbd, act_bst_accu, best_exist = False):
    
        #--------------------------------------------------------------------------#
        # train process                                                            #
        #--------------------------------------------------------------------------#
            
        model.train()
        train_loss = 0.0
        orig_train_losses = torch.zeros([num_tasks, 1])
        contrs_wc_after = torch.zeros([num_tasks, 1])
        
        cbatch_ind = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            #if batch_idx in list_batch:
                cbatch_ind += 1
                all_targets = []
                for j in range(num_tasks-1):
                    target_i = target[j] 
                    all_targets.append(target_i)
                all_targets = torch.stack(all_targets).transpose(0, 1)
                data, targets = data.to(device), all_targets.to(device, dtype=torch.int64)

                model.zero_grad()
                outputs = model(data)
                sparsity_objective = model.obj_sparsity()
                CONTRS, indiv_orig_LOSSES = contraints(model, w, a, epsilon, device, outputs,
                                                       targets, criterion, sparsity_objective)
                contrs_wc_after = contrs_wc_after + CONTRS.detach().cpu()
                orig_train_losses = orig_train_losses + indiv_orig_LOSSES
                model.zero_grad()
                loss = Augmented_Lagrangian_criterion(model, device, CONTRS, mu, lmbd)
                torch.autograd.set_detect_anomaly(True)
                loss.backward()
                optimizer.step()

                train_loss = train_loss + loss.item()
        
                if cbatch_ind % 50 == 0:
                    print('[BATCH ({}) ({:.0f}%)]\tLoss: {:.6f}'.format(
                                batch_idx+1, 100. * cbatch_ind / len(train_loader),  loss.item()))
            
        train_loss = train_loss/len(train_loader) 
        
        ################################################
        if params_init["w"][0] > 0:
            if params_init["Sparsity_study"]:
                    if params_init["is_search"]:
                        # Apply the proximity operator
                        print("Applying GrOWL ....")
                        apply_growl(model)
                        print("Done !")
                    if not params_init["is_search"]:
                        # Force parameter sharing
                        print("Forcing parameter sharing....")
                        force_sharing(model, zero_layers = params_init["zero_layers"],
                                                similarity_m = params_init["similarity_m"])
                        print("Done !")
            else:
                keep_sparse(model, params_init["info_sparse_model"])
                print("Model kept sparse !")
                
        ################################################
    
        #--------------------------------------------------------------------------#
        # Evaluation process                                                             #
        #--------------------------------------------------------------------------#
        model.eval()
        val_CORR = torch.zeros(num_tasks-1)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                all_targets = []
                for j in range(num_tasks-1):
                    target_i = target[j]
                    all_targets.append(target_i)
                all_targets = torch.stack(all_targets).transpose(0, 1)
                data, targets = data.to(device), all_targets.to(device, dtype=torch.int64)

                outputs = model(data)
                actu_val_CORR = []
                for  j in range(num_tasks-1):
                    val_pred_j = outputs[j].argmax(dim=1, keepdim=True)
                    val_corr_j = val_pred_j.eq(targets[:, j].view_as(val_pred_j)).sum().item()
                    actu_val_CORR.append(val_corr_j)
                actu_val_CORR = torch.tensor(actu_val_CORR)
                val_CORR = val_CORR + actu_val_CORR

        #--------------------------------------------------------------------------#
        # output                                                                   #
        #--------------------------------------------------------------------------#
        val_accuracy = 100 * val_CORR / len(val_loader.dataset)
        if best_exist:
            act_bst_accu = max([act_bst_accu, val_accuracy.mean().item()])
            print('\nValidation set: Average Accuracy: {:.2f}%    (Best: {:.2f}%)\n'.format(
                    val_accuracy.mean(), act_bst_accu))
        else:
            print('\nValidation set: Average Accuracy: ({:.2f}%)\n'.format(
                    val_accuracy.mean()))
            
        return orig_train_losses/len(train_loader.dataset), train_loss, val_accuracy, contrs_wc_after
    
    
