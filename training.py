import os.path
import torch
import torch.optim as optim
import datagen
import evalfun
import lossfun

DATADIR = os.path.abspath("scenario_data")
ROOTDIR = os.path.abspath("results")


def training_simple(run_name, training_data_set_name, test_data_set_name, nn_model_class, nn_model_kw, num_epochs, batch_size, opt_kw, sched_kw, loss_type="unsuper"):
    # report_path = os.path.join(ROOTDIR, run_path + ".pt")
    print("Starting run {}".format(run_name))
    report = {"run_name": run_name,
              "model_class": nn_model_class.__name__,
              "model_kw": nn_model_kw,
              "training_data_set": training_data_set_name,
              # "training_data": training_data,
              # "test_data": test_data,
              "num_epochs": num_epochs,
              "batch_size": batch_size,
              "opt_kw": opt_kw,
              "sched_kw": sched_kw,
              "loss_type": loss_type}

    print("Loading data")
    training_data = torch.load(os.path.join(DATADIR, training_data_set_name + ".pt"))
    test_data = torch.load(os.path.join(DATADIR, test_data_set_name + ".pt"))

    print("Initializing model")
    nn_model = nn_model_class(**nn_model_kw)  # creates model
    optimizer = optim.AdamW(nn_model.parameters(), **opt_kw)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sched_kw)
    num_train_samples = training_data["batch_size"]
    num_mbatches_epoch = num_train_samples // batch_size  # number of mini batches per epoch

    # for t in range (max_sgd_steps):

    training_loss = []
    # test_unsuper_loss = []
    test_loss = []
    test_anomaly_l1 = []
    test_regularizer = []
    # test_acc = []
    # test_pd = []
    detector_eval = []

    reg_param = {"lam": [], "mu": []}
    reg_param_ll = []

    for epoch in range(num_epochs+1):  # in epoch=0, only evaluation is done
        print("### Epoch {} ###".format(epoch))

        # decide loss
        loss_option_current = get_loss_option_current_epoch(loss_type, epoch)

        if epoch != 0:
            nn_model.train()
            # Shuffle the training data and labels
            permuted_indices = torch.randperm(num_train_samples)

            # Loop over mini batches
            for i_batch in range(num_mbatches_epoch):
                print("Step {}".format(i_batch+1))
                # Get the mini batch data and labels
                batch_start = i_batch * batch_size
                batch_end = (i_batch + 1) * batch_size
                mbatch_indices = permuted_indices[batch_start:batch_end]
                mbatch_data = datagen.nw_scenario_subset(training_data, mbatch_indices)

                optimizer.zero_grad()

                # with torch.autograd.detect_anomaly():
                if True:
                    model_out_P, model_out_Q, model_out_A = nn_model(mbatch_data)

                    reg_param_ll.append(nn_model.layers[-1].get_regularization_parameters(clone=True))
                    reg_param_ll_temp = nn_model.layers[-1].get_regularization_parameters()

                    loss = lossfun.lossfun(mbatch_data, model_out_P, model_out_Q, model_out_A, reg_param_ll_temp, epoch,
                                           option=loss_option_current)

                    training_loss.append(loss.detach())                    # print("train loss",train_loss)
                    loss.backward()  #
                # for n,p in nn_model.named_parameters():
                #     print(n,p)
                optimizer.step()  # performs one optimizer step, i.e., a gradient step on managed parameters

            scheduler.step()  # Update per epoch

        nn_model.eval()
        with torch.no_grad():
            model_out_P, model_out_Q, model_out_A = nn_model(test_data)

            reg_param_ll_temp = nn_model.layers[-1].get_regularization_parameters()

            loss = lossfun.lossfun(test_data, model_out_P, model_out_Q, model_out_A, reg_param_ll_temp, epoch,
                                   option=loss_option_current)

            test_loss.append(loss)
            test_anomaly_l1.append(evalfun.anomaly_l1(test_data, model_out_A))

            # det_eval = evalfun.detector_single_class(test_data, model_out_A)
            det_eval = dict()
            det_eval["auc"] = evalfun.detector_single_class_auc_approx(test_data, model_out_A[-1])["auc"]
            det_eval["auc-2"] = evalfun.detector_single_class_auc_approx(test_data, model_out_A[-2])["auc"]
            detector_eval.append(det_eval)

            reg_param["lam"].append(nn_model.lam_val)
            reg_param["mu"].append(nn_model.mu_val)

    reg_param_ll.append(nn_model.layers[-1].get_regularization_parameters(clone=True))

    training_loss = torch.stack(training_loss)
    # test_unsuper_loss = torch.stack(test_unsuper_loss)
    test_loss = torch.stack(test_loss)
    test_anomaly_l1 = torch.stack(test_anomaly_l1)
    test_regularizer = torch.stack(test_regularizer)

    report["training_loss"] = training_loss
    # report["test_unsuper_loss"] = test_unsuper_loss
    report["test_loss"] = test_loss
    report["test_anomaly_l1"] = test_anomaly_l1
    report["test_regularizer"] = test_regularizer

    report["model_dict"] = nn_model.state_dict()
    report["opt_dict"] = optimizer.state_dict()
    report["sched_dict"] = optimizer.state_dict()

    report["test_det_eval"] = {}
    for k in detector_eval[0].keys():
        report["test_det_eval"][k] = torch.stack([detector_eval[e][k] for e in range(len(detector_eval))])

    report["reg_param_ll"] = {}
    for k in reg_param_ll[0].keys():
        report["reg_param_ll"][k] = torch.stack([reg_param_ll[e][k] for e in range(len(reg_param_ll))])

    reg_param["lam"] = torch.stack(reg_param["lam"])
    reg_param["mu"] = torch.stack(reg_param["mu"])

    report["reg_param"] = reg_param

    return report


def training_simple_new(run_name, training_data_set_name, test_data_set_name, nn_model_class, nn_model_kw, num_epochs,
                    batch_size, opt_kw, sched_kw, loss_type="unsuper", fixed_steps_per_epoch=None):
    # report_path = os.path.join(ROOTDIR, run_path + ".pt")
    print("Starting run {}".format(run_name))
    report = {"run_name": run_name,
              "model_class": nn_model_class.__name__,
              "model_kw": nn_model_kw,
              "training_data_set": training_data_set_name,
              "test_data_set": test_data_set_name,
              "num_epochs": num_epochs,
              "batch_size": batch_size,
              "opt_kw": opt_kw,
              "sched_kw": sched_kw,
              "loss_type": loss_type,
              "fixed_steps_per_epoch": fixed_steps_per_epoch}

    print("Loading data")
    training_data = torch.load(training_data_set_name + ".pt")
    test_data = torch.load(test_data_set_name + ".pt")

    print("Initializing model")
    nn_model = nn_model_class(**nn_model_kw)  # creates model
    optimizer = optim.AdamW(nn_model.parameters(), **opt_kw)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **sched_kw)
    num_train_samples = training_data["batch_size"]
    if fixed_steps_per_epoch:
        num_mbatches_epoch = fixed_steps_per_epoch
    else:
        num_mbatches_epoch = num_train_samples // batch_size  # number of mini batches per epoch

    # for t in range (max_sgd_steps):

    training_loss = []
    test_loss = []
    test_anomaly_l2 = []
    detector_eval = []

    reg_param = {"lam": [], "mu": []}
    reg_param_ll = []

    for epoch in range(num_epochs + 1):  # in epoch=0, only evaluation is done
        print("### Epoch {} ###".format(epoch))

        # decide loss
        loss_option_current = get_loss_option_current_epoch(loss_type, epoch)

        if epoch != 0:
            nn_model.train()
            # Shuffle the training data and labels
            permuted_indices = torch.randperm(num_train_samples)

            # Loop over mini batches
            for i_batch in range(num_mbatches_epoch):
                print("Step {}".format(i_batch + 1))
                # Get the mini batch data and labels
                if fixed_steps_per_epoch and fixed_steps_per_epoch > num_train_samples // batch_size:
                    perm_idx_batch_idx = list(crange(i_batch * batch_size, (i_batch + 1) * batch_size, num_train_samples))
                    mbatch_indices = permuted_indices[perm_idx_batch_idx]
                else:
                    batch_start = i_batch * batch_size
                    batch_end = (i_batch + 1) * batch_size
                    mbatch_indices = permuted_indices[batch_start:batch_end]
                mbatch_data = datagen.nw_scenario_subset(training_data, mbatch_indices)

                optimizer.zero_grad()

                # with torch.autograd.detect_anomaly():
                if True:
                    model_out_P, model_out_Q, model_out_A = nn_model(mbatch_data)

                    reg_param_ll.append(nn_model.layers[-1].get_regularization_parameters(clone=True))
                    reg_param_ll_temp = nn_model.layers[-1].get_regularization_parameters()

                    loss = lossfun.lossfun(mbatch_data, model_out_P, model_out_Q, model_out_A, reg_param_ll_temp, epoch,
                                           option=loss_option_current)

                    training_loss.append(loss.detach())
                    loss.backward()  #
                # for n,p in nn_model.named_parameters():
                #     print(n,p)
                optimizer.step()  # performs one optimizer step, i.e., a gradient step on managed parameters

            scheduler.step()  # Update per epoch

        nn_model.eval()
        with torch.no_grad():
            model_out_P, model_out_Q, model_out_A = nn_model(test_data)

            reg_param_ll_temp = nn_model.layers[-1].get_regularization_parameters()

            loss = lossfun.lossfun(test_data, model_out_P, model_out_Q, model_out_A, reg_param_ll_temp, epoch,
                                   option=loss_option_current)

            test_loss.append(loss)
            test_anomaly_l2.append(evalfun.anomaly_l2(test_data, model_out_A))

            det_eval = dict()
            det_eval["auc"] = evalfun.detector_single_class_auc_approx(test_data, model_out_A[-1])["auc"]
            det_eval["auc-2"] = evalfun.detector_single_class_auc_approx(test_data, model_out_A[-2])["auc"]
            detector_eval.append(det_eval)

            reg_param["lam"].append(nn_model.lam_val)
            reg_param["mu"].append(nn_model.mu_val)

    reg_param_ll.append(nn_model.layers[-1].get_regularization_parameters(clone=True))

    training_loss = torch.stack(training_loss)
    test_loss = torch.stack(test_loss)
    test_anomaly_l2 = torch.stack(test_anomaly_l2)

    report["training_loss"] = training_loss
    report["test_loss"] = test_loss
    report["test_anomaly_l2"] = test_anomaly_l2

    report["model_dict"] = nn_model.state_dict()
    report["opt_dict"] = optimizer.state_dict()
    report["sched_dict"] = optimizer.state_dict()

    report["test_det_eval"] = {}
    for k in detector_eval[0].keys():
        report["test_det_eval"][k] = torch.stack([detector_eval[e][k] for e in range(len(detector_eval))])

    report["reg_param_ll"] = {}
    for k in reg_param_ll[0].keys():
        report["reg_param_ll"][k] = torch.stack([reg_param_ll[e][k] for e in range(len(reg_param_ll))])

    reg_param["lam"] = torch.stack(reg_param["lam"])
    reg_param["mu"] = torch.stack(reg_param["mu"])

    report["reg_param"] = reg_param

    return report


def get_loss_option_current_epoch(loss_type, epoch):
    # loss_type either string or dict with thresholds-> loss_type
    # In case of curriculum, applies loss_type if epoch >= threshold

    if isinstance(loss_type, dict):
        max_threshold = -1
        loss_type_current = None
        for threshold, loss in loss_type.items():
            if max_threshold < threshold <= epoch:
                max_threshold = threshold
                loss_type_current = loss
    else:
        loss_type_current = loss_type

    return loss_type_current


def crange(start, end, modulo):
    for i in range(start,end):
        yield i % modulo