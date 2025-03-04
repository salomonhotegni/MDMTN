import numpy as np
import torch
from sklearn.cluster import AffinityPropagation

from src.utils.projectedOWL_utils import proxOWL

#######################################
#####  Helper functions for GrOWL #####
#######################################

# Generate GrOWL coefficients
def create_GrOWL_params(model, n):
        beta1 = model.GrOWL_parameters["beta1"]
        beta2 = model.GrOWL_parameters["beta2"]
        tp = model.GrOWL_parameters["tp"]
        if tp == "spike":
            teta_is = [beta1 + beta2] + [beta2]*(n-1)
            return teta_is
        if tp == "linear":
            teta_is = [beta1 + beta2*(n-i)/n for i in range(1, n+1)]
            return teta_is
        if tp == "Dejiao":
            p = int(n/2)
            teta_is_1 = [beta1 + beta2*(p-i+1) for i in range(1, p+1)]
            teta_is_2 = [beta2]*(n-p)
            teta_is = teta_is_1 + teta_is_2
            return teta_is
        else:
            raise ValueError("Type should be 'spike', 'linear', or 'Dejiao'!")
        ###############################         
            
def apply_growl(model,):
            model_weights = model.state_dict()
            pl_srate = model.GrOWL_parameters["max_layerSRate"]
            for name, weight in model_weights.items():
                if ('weight' in name) and (len(weight.shape)>1):
                    # Reshape the weight tensor
                    # if conv: lx(l-1)xwxh ----> (l-1)x(l.w.h)
                    # if fc: lx(l-1) ----> (l-1)xl
                    # Therefore: Weight matrix's ROWS ==> previous layer's NEURONS
                    org_shape = weight.shape
                    if ("task_blocks" in name) or (org_shape[1] == model.GrOWL_parameters["skip_layer"]): 
                        continue
                    
                    if len(org_shape) == 2:
                        reshaped_weight = weight.T
                    else:
                        reshaped_weight = weight.view(weight.shape[1], -1)
                    
                    # Compute the norm-2 of each row
                    n2_rows_W = torch.norm(reshaped_weight, p=2, dim=1)


                    ##########################################
                    # Create GrOWL params
                    # _, s_inds = torch.sort(n2_rows_W)
                    # theta_is = create_GrOWL_params(model, len(s_inds))
                    theta_is = create_GrOWL_params(model, len(n2_rows_W))
                    # sorted_theta_is = [theta_is[i] for i in torch.argsort(s_inds)]

                    # Prox operator
                    # new_n2_rows_W = proxOWL(n2_rows_W.cpu().numpy(), np.array(sorted_theta_is))
                    new_n2_rows_W = proxOWL(n2_rows_W.cpu().numpy(), np.array(theta_is))

                    
                    new_W = torch.zeros_like(reshaped_weight)
                    eps = torch.finfo(torch.float32).eps
                    for i in range(reshaped_weight.shape[0]):
                        if n2_rows_W[i] < eps:
                            new_W[i, :] = torch.zeros_like(reshaped_weight[i, :])
                        else:
                            new_W[i, :] = reshaped_weight[i, :] * (new_n2_rows_W[i] / n2_rows_W[i])
                    ############################################

                    # if applying GrOWL leads to more than "pl_srate %" of zero rows:
                    zero_row_idcs = torch.nonzero((new_W == 0).all(dim=1)).squeeze()
                    max_slct = int(pl_srate*new_W.shape[0])
                    if (torch.numel(zero_row_idcs) > max_slct):
                        # use_slct = new_W.shape[0]-max_slct
                        numel = zero_row_idcs.numel()
                        use_slct = numel-max_slct
                        shuffled_idcs = torch.randperm(numel)
                        selected_idcs = shuffled_idcs[:use_slct]
                        selected_elmts = zero_row_idcs.flatten()[selected_idcs]
                        new_W[selected_elmts] = reshaped_weight[selected_elmts] # use original values
                    
                    # Reshape new_W back to its original shape
                    if len(org_shape) == 2:
                        new_W = new_W.T
                    else:
                        new_W = new_W.view(org_shape)
                        
                    # Update weight
                    weight.copy_(new_W)
            ###############################

def similarity_info(model, zero_layers):
        with torch.no_grad():
            similarity_m = {}
            sm_p = model.GrOWL_parameters["sim_preference"]
            for name, weight in model.named_parameters():
                if (name not in zero_layers) and ('weight' in name) and (len(weight.shape)>1):
                    # Reshape the weight tensor
                    # if conv: lx(l-1)xwxh ----> (l-1)x(l.w.h)
                    # if fc: lx(l-1) ----> (l-1)xl
                    # Therefore: Weight matrix's ROWS ==> previous layer's NEURONS
                    org_shape = weight.shape
                    if ("task_blocks" in name) or (org_shape[1] == model.GrOWL_parameters["skip_layer"]): 
                        continue
                    if len(org_shape) == 2:
                        reshaped_weight = weight.T
                    else:
                        reshaped_weight = weight.view(weight.shape[1], -1)
                    
                    #n, p = reshaped_weight.shape
                    p_scl = torch.matmul(reshaped_weight, reshaped_weight.T)
                    i_norm = torch.norm(reshaped_weight, p=2, dim=1)**2
                    j_norm = i_norm.view(-1, 1)
                    sim_matrix = p_scl / torch.max(i_norm, j_norm)

                    # NaN values occur when the weight contains "zero lines"
                    # Replace them by 1.0 ====> "zero rows" are perfectly similar
                    # to belong to the same group
                    sim_matrix = torch.nan_to_num(sim_matrix, nan=1.0)
                    
                    OWL_clustering = AffinityPropagation(affinity='precomputed', preference =sm_p)
                    # OWL_clustering.fit(sim_matrix)
                    # labels = OWL_clustering.labels_
                    labels = OWL_clustering.fit_predict(sim_matrix.cpu())
                    # Get the indices of the cluster centers
                    cl_centers_idx = OWL_clustering.cluster_centers_indices_

                    # Identify zeros rows:
                    zeros_rows = torch.sum(reshaped_weight, dim=1) == 0
                    zeros_rows = zeros_rows.nonzero(as_tuple=False).squeeze().tolist()

                    similarity_m[name] = [sim_matrix, cl_centers_idx, labels, zeros_rows]
            
            return similarity_m
        #################################
            
def sparsity_info(model, verbose = True):
        sparsity_ratio = 0.0
        total_ns = 0.0
        zero_layers = []
        with torch.no_grad():
            for name, weight in model.named_parameters():
                if ('weight' in name) and (len(weight.shape)>1):
                    if len(weight.shape) == 2:
                        reshaped_weight = weight.T
                    else:   
                        reshaped_weight = weight.view(weight.shape[1], -1)
                        
                    k = torch.count_nonzero(torch.sum(reshaped_weight, dim=1) == 0).item()

                    sparsity_ratio = sparsity_ratio + k
                    total_ns = total_ns + reshaped_weight.shape[0]
                    if verbose:
                        print("Name: ", name)
                        print(f"Insignificant Neurons: {k}/{reshaped_weight.shape[0]} ({100*k/reshaped_weight.shape[0]})")
                        print("====================================")
                        
                    if torch.all(reshaped_weight == 0):
                        zero_layers.append(name)
        
        sparsity_ratio = 100*sparsity_ratio/total_ns
        print("Sparsity Ratio: ", sparsity_ratio)
        return sparsity_ratio, zero_layers
        #######################
        
            
def force_sharing(model, zero_layers, similarity_m):
        model_weights = model.state_dict()
        for name, weight in model_weights.items():
            if (name not in zero_layers) and ('weight' in name) and (len(weight.shape)>1):
                # Reshape the weight tensor
                # if conv: lx(l-1)xwxh ----> (l-1)x(l.w.h)
                # if fc: lx(l-1) ----> (l-1)xl
                # Therefore: Weight matrix's ROWS ==> previous layer's NEURONS
                org_shape = weight.shape
                if ("task_blocks" in name) or (org_shape[1] == model.GrOWL_parameters["skip_layer"]): 
                    continue
                    
                _, cl_centers_idx, labels, zeros_rows = similarity_m[name]
                
                if len(org_shape) == 2:
                    reshaped_weight = weight.T
                else:
                    reshaped_weight = weight.view(weight.shape[1], -1)
                
                reshaped_weight[zeros_rows] = 0

                for i in range(len(cl_centers_idx)):
                    cl_data_pts = torch.where(torch.from_numpy(labels) == i)[0]
                    centroid = torch.mean(reshaped_weight[cl_data_pts], dim=0)
                    reshaped_weight[cl_data_pts] = centroid
                
                # Reshape reshaped_weight back to its original shape
                if len(org_shape) == 2:
                    reshaped_weight = reshaped_weight.T
                else:
                    reshaped_weight = reshaped_weight.view(org_shape)
                # Update weight
                weight.copy_(reshaped_weight)
        ######################################
        
def get_sparse_model_info(model,):
        with torch.no_grad():
            info_sparse_model = {}
            for name, weight in model.named_parameters():
                if 'weight' in name and (len(weight.shape)>1):
                    # Reshape the weight tensor
                    # if conv: lx(l-1)xwxh ----> (l-1)x(l.w.h)
                    # if fc: lx(l-1) ----> (l-1)xl
                    # Therefore: Weight matrix's ROWS ==> previous layer's NEURONS
                    org_shape = weight.shape
                    if len(org_shape) == 2:
                        reshaped_weight = weight.T
                    else:
                        reshaped_weight = weight.view(weight.shape[1], -1)
                    
                    # Identify zeros rows:
                    zeros_rows = torch.sum(reshaped_weight, dim=1) == 0
                    zeros_rows = zeros_rows.nonzero(as_tuple=False).squeeze().tolist()

                    info_sparse_model[name] = zeros_rows
            
            return info_sparse_model
        #################################
        
def keep_sparse(model, info_sparse_model):
        model_weights = model.state_dict()
        for name, weight in model_weights.items():
            if 'weight' in name and (len(weight.shape)>1):
                # Reshape the weight tensor
                # if conv: lx(l-1)xwxh ----> (l-1)x(l.w.h)
                # if fc: lx(l-1) ----> (l-1)xl
                # Therefore: Weight matrix's ROWS ==> previous layer's NEURONS
                org_shape = weight.shape

                zeros_rows = info_sparse_model[name]
                
                if len(org_shape) == 2:
                    reshaped_weight = weight.T
                else:
                    reshaped_weight = weight.view(weight.shape[1], -1)
                
                reshaped_weight[zeros_rows] = 0
                
                # Reshape reshaped_weight back to its original shape
                if len(org_shape) == 2:
                    reshaped_weight = reshaped_weight.T
                else:
                    reshaped_weight = reshaped_weight.view(org_shape)
                # Update weight
                weight.copy_(reshaped_weight)
        ######################################
        
def metrics_tr(model, verbose = True):
        sparsity_ratio = 0.0
        compr_ratio = 1.0
        params_shg = 1.0
        
        total_ns = 0.0
        zeros_ns = 0.0
        nonzeros_ns = 0.0
        unique_ns = 0.0
        zero_layers = []
        with torch.no_grad():
            for name, weight in model.named_parameters():
                if ('weight' in name) and (len(weight.shape)>1):
                    #print(weight)
                    if len(weight.shape) == 2:
                        reshaped_weight = weight.T
                    else:   
                        reshaped_weight = weight.view(weight.shape[1], -1)
                        
                    r = torch.count_nonzero(torch.sum(reshaped_weight, dim=1) == 0).item()
                            
                    rshpd_wght_np = reshaped_weight.cpu().numpy()
                    unique_rows = np.unique(rshpd_wght_np, axis=0)
                    
                    unique_ns = unique_ns + len(unique_rows)
                    total_ns = total_ns + reshaped_weight.shape[0]
                    nonzeros_ns = nonzeros_ns + (reshaped_weight.shape[0] - r)
                    zeros_ns = zeros_ns + r

                    if verbose:
                        print("Name: ", name)
                        print(f"Insignificant: {r}/{reshaped_weight.shape[0]} ({100*r/reshaped_weight.shape[0]} %)")
                        print("====================================")
                        
                    if torch.all(reshaped_weight == 0):
                        zero_layers.append(name)
        
        sparsity_ratio = 100*zeros_ns/total_ns
        compr_ratio = total_ns/unique_ns
        params_shg = nonzeros_ns/unique_ns
        print(" ####### Training Results ####### ")
        print("Sparsity Rate: ", sparsity_ratio)
        print("Compression Rate: ", compr_ratio)
        print("Parameter Sharing: ", params_shg)
        print(" ################################ ")
        return [sparsity_ratio, compr_ratio, params_shg], zero_layers
        ######################################
            
