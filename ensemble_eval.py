import torch
from timeit import default_timer as timer
import numpy as np
import os


def simple_p_comb(tcs, comb_m=0, _=0, weights=None):
    if comb_m == 0:
        ps = torch.sum(tcs, dim=0) / len(tcs)
        return ps
    elif comb_m == 1:
        ps = torch.median(tcs.cuda(), dim=0).values
        ps = ps / torch.sum(ps, dim=1).unsqueeze(1).expand(ps.size())
        return ps.cpu()
    elif comb_m == 2:
        c, n, k = tcs.size()
        w = weights.unsqueeze(1).unsqueeze(2).expand(c, n, k)
        ps = torch.sum(w*tcs, dim=0)
        return ps


def m1_lin_multi_auto_batch_c(tcs, comb_m=0, batch_decr=0, weights=None):
    """First index of tcs is index of source to combine, second is sample index and third is label index"""
    # c-number of combined inputs, n-number of samples, k-length of prob vector
    c, n, k = tcs.size()
    s = tcs[0, 0].element_size()
    all_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    batch_size = int((1 - 0.1*batch_decr) *
                     ((all_mem - s*k*k)/(s*k*((5 + (comb_m == 2))*c*k+7*k + 5))))
    # print('S: ' + str(all_mem) + ', s: ' + str(s) + ', c: ' + str(c) + ', k: ' + str(k))
    # print(torch.cuda.memory_summary(0))
    print("Batch size: " + str(batch_size))

    bn = 0
    # constants preparation on gpu
    E = torch.eye(k).cuda()
    VE = (1 - E).unsqueeze(0).unsqueeze(1).expand(c, batch_size, k, k)
    Es = E.unsqueeze(0).expand(batch_size, k, k)
    B = torch.zeros(batch_size, k, 1).cuda()
    B[:, k - 1, :] = 1
    if comb_m == 2:
        w = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(c, batch_size, k, k).cuda()

    ps_all = torch.tensor([], dtype=torch.get_default_dtype())

    for si in range(0, n, batch_size):
        print('Batch: ' + str(bn))
        # print(torch.cuda.memory_summary(0))
        tcsp = tcs[:, si: si + batch_size, :].cuda()

        # last batch may have smaller size
        if tcsp.size()[1] < batch_size:
            VE = VE[:, 0:tcsp.size()[1], :, :]
            Es = Es[0:tcsp.size()[1], :, :]
            B = B[0:tcsp.size()[1], :, :]
            if comb_m == 2:
                w = w[:, 0:tcsp.size()[1], :, :]

        # four dimensional tensor, first d - input index, second d - index of sample in batch, last two d - matrices
        # with columns filled by respective prob vector and zero diagonal
        TCs = VE*tcsp.unsqueeze(3)

        if comb_m == 0:
            # R as an average of pairwise probability matrices for respective inputs
            R = torch.sum(TCs / (TCs + TCs.transpose(2, 3) + (TCs == 0)), dim=0) / c
        elif comb_m == 1:
            # R as an median of pairwise probability matrices for respective inputs
            R = torch.median(TCs / (TCs + TCs.transpose(2, 3) + (TCs == 0)), dim=0).values
        elif comb_m == 2:
            # R as an weighted sum of pairwise probability matrices for respective inputs
            R = torch.sum(w * (TCs / (TCs + TCs.transpose(2, 3) + (TCs == 0))), dim=0)

        # method 1
        A = (R.sum(dim=2).diag_embed() + R)/(k - 1) - Es
        A[:, k - 1, :] = 1

        Xs, LUs = torch.solve(B, A)
        ps = Xs[:, 0:k, 0:1].squeeze(2)
        torch.cuda.empty_cache()
        ps_all = torch.cat((ps_all, ps.cpu()), 0)
        bn += 1

        # print(torch.cuda.memory_summary(0))

    del E, VE, Es, B, tcsp, TCs, R, A, Xs, LUs, ps

    return ps_all


def m2_lin_multi_auto_batch_c(tcs, comb_m=0, batch_decr=0, weights=None):
    """First index of tcs is index of source to combine, second is sample index and third is label index"""
    c, n, k = tcs.size()
    s = tcs[0, 0].element_size()
    all_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    batch_size = int((1 - 0.1*batch_decr) *
                     ((all_mem - s * k * k) / (((4 +(comb_m == 2))*c*k*k + 4*k*k + k * 3*c*k + 3*k*k + 6*(k+1)**2 + 6*(k + 1) + 3)*s)))


    print("Batch size: " + str(batch_size))
    # c-number of combined inputs, n-number of samples, k-length of prob vector
    c, n, k = tcs.size()
    bn = 0
    # constants preparation on gpu
    E = torch.eye(k).cuda()
    VE = (1 - E).unsqueeze(0).unsqueeze(1).expand(c, batch_size, k, k)
    es = torch.ones(batch_size, k, 1, dtype=torch.get_default_dtype()).cuda()
    B = torch.zeros(batch_size, k + 1, 1).cuda()
    B[:, k, :] = 1
    zs = torch.zeros(batch_size, 1, 1).cuda()
    if comb_m == 2:
        w = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(c, batch_size, k, k).gpu()

    ps_all = torch.tensor([], dtype=torch.get_default_dtype())

    for si in range(0, n, batch_size):
        print('Batch: ' + str(bn))
        #print(torch.cuda.memory_summary(0))
        tcsp = tcs[:, si: si + batch_size, :].cuda()

        # last batch may have smaller size
        if tcsp.size()[1] < batch_size:
            VE = VE[:, 0:tcsp.size()[1], :, :]
            es = es[0:tcsp.size()[1], :, :]
            B = B[0:tcsp.size()[1], :, :]
            zs = zs[0:tcsp.size()[1], :, :]
            if comb_m == 2:
                w = w[:, 0:tcsp.size()[1], :, :]

        # four dimensional tensor, first d - input index, second d - index of sample in batch, last two d - matrices
        # with columns filled by respective prob vector and zero diagonal
        TCs = VE*tcsp.unsqueeze(3)

        if comb_m == 0:
            # R as an average of conditional sum matrices for respective inputs
            R = torch.sum(TCs / (TCs + TCs.transpose(2, 3) + (TCs == 0)), dim=0)/c
        elif comb_m == 1:
            # R as an median of conditional sum matrices for respective inputs
            R = torch.median(TCs / (TCs + TCs.transpose(2, 3) + (TCs == 0)), dim=0).values
        elif comb_m == 2:
            # R as an weighted sum of pairwise probability matrices for respective inputs
            R = torch.sum(w * (TCs / (TCs + TCs.transpose(2, 3) + (TCs == 0))), dim=0)

        # method 2
        Q = (R * R).sum(dim=1).diag_embed() - R * R.transpose(1, 2)
        A = torch.cat((Q, es), 2)
        A = torch.cat((A, torch.cat((es.transpose(1, 2), zs), 2)), 1)
        Xs, LUs = torch.solve(B, A)
        ps = Xs[:, 0:k, 0:1].squeeze(2)
        # torch.cuda.empty_cache()
        ps_all = torch.cat((ps_all, ps.cpu()), 0)
        bn += 1

    del E, VE, es, B, tcsp, TCs, R, A, Xs, LUs, ps

    return ps_all


def compute_acc_topk(y_cor, ps, l):
    top_v, top_i = torch.topk(ps, l, dim=1)
    n = y_cor.size()[0]

    return torch.sum(top_i == y_cor.unsqueeze(1)).item() / n


def eval_ensembles(fold, out_name, method_id, comb_id, output_probs_fold=None):
    """ fold: folder containing models.csv, y_val.npy and outputs of neural networks
        out_name: name of output csv file
        method_id: 0 - simple probability combining, 1 - method 1, 2 - method 2
        comb_id: 0 - average, 1 - median
        output_probs_file: if not null, resulting probabilities will be saved into this folder """
    models_file = os.path.join(fold, 'models.csv')
    y_file = os.path.join(fold, 'y_val.npy')
    output_file = os.path.join(fold, out_name)

    print('Reading models file')
    models = []
    m_file = open(models_file, 'r')
    header = m_file.readline()[:-1].split(',')
    for line in m_file:
        models.append([int(e) for e in line[:-1].split(',')])

    m_file.close()

    o_file = open(output_file, 'w')
    o_file.write('rowid,k,method,top1,top5,time\n')
    o_file.close()

    print('Reading neur outputs')
    neur_names = header[2:]
    tcs = []
    for m in neur_names:
        tcs.append(torch.tensor(np.load(os.path.join(fold, m + '.npy'))).unsqueeze(0))

    neur_ps = torch.cat(tcs, 0)

    methods = [simple_p_comb, m1_lin_multi_auto_batch_c, m2_lin_multi_auto_batch_c]
    method_acronyms = ['p', 'm1', 'm2']
    comb_acronyms = ['avg', 'median', 'weighted']

    print('Reading correct labels')
    y_cor = torch.tensor(np.load(y_file))

    for mod in models:
        print('Computing rowid: ' + str(mod[0]))
        comb_time = 0
        neur_ps_subset = neur_ps[[nn == 1 for nn in mod[2:]]]

        time_s = timer()
        fin = False
        tries = 0
        while not fin and tries < 10:
            if tries > 0:
                torch.cuda.empty_cache()
                print('Trying again: ' + str(tries))
            try:
                ps = methods[method_id](neur_ps_subset, comb_id, tries)
                fin = True
            except RuntimeError as rerr:
                if 'memory' not in str(rerr):
                    raise rerr
                print("OOM Exception")
                del rerr
                tries += 1

        if not fin:
            print('Unsuccessful')
            return -1

        top1 = compute_acc_topk(y_cor, ps, 1)
        top5 = compute_acc_topk(y_cor, ps, 5)

        o_file = open(output_file, 'a')
        o_file.write(str(mod[0]) + ',' + str(mod[1]) + ',' + method_acronyms[method_id] + '_' + comb_acronyms[comb_id]
                     + ',' + str(top1) + ',' + str(top5) + ',' + str(timer() - time_s) + '\n')
        o_file.close()

        if output_probs_fold is not None:
            out_file_n = 'rowid_' + str(mod[0]) + '_k_' + str(mod[1]) + '_' + \
                         method_acronyms[method_id] + '_' + comb_acronyms[comb_id] + '.npy'
            np.save(os.path.join(output_probs_fold, out_file_n), ps)

        comb_time += timer() - time_s

    print('Finished in[s]: ' + str(comb_time))


def eval_weighted_ensemble(fold, out_name, method_id, output_probs_fold=None):
    """ fold: folder containing models.csv, y_val.npy and outputs of neural networks
        out_name: name of output csv file
        method_id: 0 - simple probability combining, 1 - method 1, 2 - method 2
        output_probs_file: if not null, resulting probabilities will be saved into this folder """
    models_file = os.path.join(fold, 'models.csv')
    y_file = os.path.join(fold, 'y_val.npy')
    output_file = os.path.join(fold, out_name)

    print('Reading models file')
    models = []
    m_file = open(models_file, 'r')
    header = m_file.readline()[:-1].split(',')
    for line in m_file:
        models.append([float(e) for e in line[:-1].split(',')])

    m_file.close()

    o_file = open(output_file, 'w')
    o_file.write('rowid,k,method,top1,top5,time\n')
    o_file.close()

    print('Reading neur outputs')
    neur_names = header[2:]
    tcs = []
    for m in neur_names:
        tcs.append(torch.tensor(np.load(os.path.join(fold, m + '.npy'))).unsqueeze(0))

    neur_ps = torch.cat(tcs, 0)

    methods = [simple_p_comb, m1_lin_multi_auto_batch_c, m2_lin_multi_auto_batch_c]
    method_acronyms = ['p', 'm1', 'm2']
    comb_acronym = 'weighted'

    print('Reading correct labels')
    y_cor = torch.tensor(np.load(y_file))

    for mod in models:
        print('Computing rowid: ' + str(int(mod[0])))
        comb_time = 0
        weights = torch.tensor(mod[2:])
        time_s = timer()
        fin = False
        tries = 0
        while not fin and tries < 10:
            if tries > 0:
                torch.cuda.empty_cache()
                print('Trying again: ' + str(tries))
            try:
                ps = methods[method_id](neur_ps, 2, tries, weights)
                fin = True
            except RuntimeError as rerr:
                if 'memory' not in str(rerr):
                    raise rerr
                print("OOM Exception")
                del rerr
                tries += 1

        if not fin:
            print('Unsuccessful')
            return -1

        top1 = compute_acc_topk(y_cor, ps, 1)
        top5 = compute_acc_topk(y_cor, ps, 5)

        o_file = open(output_file, 'a')
        o_file.write(str(int(mod[0])) + ',' + str(int(mod[1])) + ',' + method_acronyms[method_id] + '_' + comb_acronym
                     + ',' + str(top1) + ',' + str(top5) + ',' + str(timer() - time_s) + '\n')
        o_file.close()

        if output_probs_fold is not None:
            out_file_n = 'rowid_' + str(int(mod[0])) + '_k_' + str(int(mod[1])) + '_' + \
                         method_acronyms[method_id] + '_' + comb_acronym + '.npy'
            np.save(os.path.join(output_probs_fold, out_file_n), ps)

        comb_time += timer() - time_s

    print('Finished in[s]: ' + str(comb_time))