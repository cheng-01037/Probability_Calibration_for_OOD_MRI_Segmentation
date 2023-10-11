"""
tunning script for the proposed method
TODO: merge with the lts script
"""
import time
import shutil
from models import create_proposed
from my_utils.util import AttrDict, worker_init_fn
import SimpleITK as sitk
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import dataloaders.niftiio as nio
from configs_exp import ex
from tensorboardX import SummaryWriter

def prediction_wrapper(model, test_loader, opt, epoch, label_name, mode = 'base', fold = -1):
    """
        when using multi inference mode,
    """
    with torch.no_grad():
        out_prediction_list = {}
        out_count = 0
        nclass = len(label_name)
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['img'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_logits = torch.Tensor(np.zeros( [nframe, nclass , nx, ny]  )).cuda()
                curr_prob = torch.Tensor(np.zeros( [nframe, nclass , nx, ny]  )).cuda()
                curr_temp = torch.Tensor(np.zeros( [nframe, nclass , nx, ny]  )).cuda()
            assert batch['lb'].shape[0] == 1 # enforce a batchsize of 1

            test_input = {
                    'img': batch['img'],
                    'lb': batch['lb']
                    }

            model.set_input(test_input)

            gth, pred, pred_logit, tempmap, _ = model.get_calib_gpu()
            curr_logits[slice_idx, ...]    = pred_logit[0, ...]
            curr_temp[slice_idx, ...]      = tempmap[0, ...]
            curr_prob[slice_idx, ...]      = F.softmax(pred_logit, dim  = 1)[0, ...]

            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]

            slice_idx += 1

            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth

                out_prediction_list[scan_id_full]['logits'] = curr_logits
                out_prediction_list[scan_id_full]['calibrated_prob'] = curr_prob
                out_prediction_list[scan_id_full]['temp_map'] = curr_temp

                out_count += 1

        print("Epoch {} test result are shown as follows:".format(epoch))
        return_which = ['dice']  + ['dilate_ece', 'dilate_scev2']

        error_dict, error_tables, domain_names = eval_list_wrapper_domain(out_prediction_list, len(label_name ), model, label_name,return_which = return_which)
        error_dict["mode"] = mode

    return out_prediction_list, error_tables, error_dict, domain_names

def eval_list_wrapper_domain(vol_list, nclass, model, label_name, return_which, confidence_key = 'calibrated_prob'):
    """
    return_which:   which metric to record, dice, prec and recall
    prob_key: which prediction map as the confidence value for evaluation calibration
    """
    out_count = len(vol_list)
    output_by_domain = {} # tables by domain
    conf_mat_list = []

    metrics_dict = {}
    for _metric in return_which:
        metrics_dict[_metric] = np.ones([ out_count, nclass ]  ) * -1.

    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in output_by_domain.keys():
            output_by_domain[domain] = {}
            for _metric in return_which:
                output_by_domain[domain][_metric] = {'scores': [],
                    'scan_ids': []}

        pred_       = comp['pred']
        gth_        = comp['gth']
        confidence_ = comp[confidence_key]
        calibrated_logits_ = comp['logits']

        seg_result_dict = model.ScoreAllEval(torch.unsqueeze(pred_, 1), gth_, dense_input = True) # segmentation errors

        dilate_evals_dict = model.dilate_calib_metric_wrapper(confidence_.permute(1,0,2,3), gth_) # evaluated on adaptive ROIs

        seg_result_dict['dilate_ece']     = np.float32([dilate_evals_dict['expected_calibration_error']])
        seg_result_dict['dilate_scev2']   = np.float32([dilate_evals_dict['static_calibration_errorv2']])

        for _metric, _score in seg_result_dict.items():
            if _metric not in set(return_which):
                continue
            output_by_domain[domain][_metric]['scores'].append( [_sc for _sc in _score ]  )
            output_by_domain[domain][_metric]['scan_ids'].append( scan_id )
            metrics_dict[_metric][idx, ...] = np.reshape(_score, (-1))
        del pred_
        del gth_
        idx += 1

    # compute results
    error_dict = {}
    for _metric in return_which:
        print(f'Evaluating metric {_metric}:')
        error_dict[_metric] = {}

        for organ in range(nclass):
            mean_met = np.mean( metrics_dict[_metric][:, organ] )
            error_dict[_metric][label_name[organ]] = mean_met
        print("Overall mean {} by sample {:06.5f} \n".format(_metric,  metrics_dict[_metric][:,1:].mean()))
        error_dict[_metric]['overall'] = metrics_dict[_metric][:,1:].mean()

    # then output and record the results categorized by domains (type of artifacts)
    overall_by_domain = {}
    domain_names = []
    for _metric in return_which:
        overall_by_domain[_metric] = []
        for domain_name, domain_dict in output_by_domain.items():
            domain_scores = np.array( output_by_domain[domain_name][_metric]['scores']  )
            if _metric in set(['dice', 'prec', 'recall']  ): # these are class-conditional whose background need to be excluded from computation
                domain_mean_score = np.mean(domain_scores[:, 1:])
            else:
                domain_mean_score = np.mean(domain_scores )
                assert domain_scores.shape[-1] < 2
            error_dict[_metric][f'domain_{domain_name}_overall'] = domain_mean_score
            error_dict[_metric][f'domain_{domain_name}_table'] = domain_scores
            overall_by_domain[_metric].append(domain_mean_score)
            domain_names.append(domain_name)
        error_dict[_metric]['overall_by_domain'] = np.mean(overall_by_domain[_metric])

    return error_dict, metrics_dict, list(set(domain_names))

@ex.automain
def main(_run, _config, _log):
    # result logging and experiment tracking
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        _config['run_dir'] = _run.observers[0].dir
        _config['snapshot_dir'] = f'{_run.observers[0].dir}/snapshots'
        _config['pred_dir'] = f'{_run.observers[0].dir}/interm_preds'

    tbfile_dir = os.path.join(  _run.observers[0].dir, 'tboard_file' ); os.mkdir(tbfile_dir)
    tb_writer = SummaryWriter( tbfile_dir  )
    opt = AttrDict(_config)

    if opt.data_name == 'ACDC': # the calibrator is still tuned on acdc, and the artifacts are assumed to be unseen to calibrator before testing
        import dataloaders.ACDCDataset as ACDC
        import dataloaders.ACDCCDataset as ACDC_C
        train_set           = ACDC.get_tuneval(opt.eval_fold )
        val_source_set      = ACDC.get_validation(opt.eval_fold)
        test_source_set     = ACDC.get_test(opt.eval_fold)
        test_ood_set        = ACDC_C.get_test(opt.eval_fold, domains = opt.acdcc_domains)
        label_name          = ACDC.LABEL_NAME

    else:
        raise NotImplementedError(opt.data_name)

    aleatrain_loader = DataLoader(dataset = train_set, num_workers = opt.nThreads,\
            batch_size = opt.batchSize, shuffle = True, drop_last = True, worker_init_fn = worker_init_fn, pin_memory = True)

    pretrain_loader = DataLoader(dataset = train_set, num_workers = opt.nThreads,\
            batch_size = opt.daebatchSize, shuffle = True, drop_last = True, worker_init_fn = worker_init_fn, pin_memory = True)

    val_loader = iter(DataLoader(dataset = val_source_set, num_workers = 1,
            batch_size = 1, shuffle = True, drop_last = True, pin_memory = True))

    test_ood_loader = DataLoader(dataset = test_ood_set, num_workers = 1,
            batch_size = 1, shuffle = False, pin_memory = True)

    test_source_loader = DataLoader(dataset = test_source_set, num_workers = 1,
            batch_size = 1, shuffle = False, pin_memory = True)

    if opt.exp_type == 'tune_calibrator':
        if opt.calib_type == 'proposed':
            model = create_proposed(opt)
        else:
            raise NotImplementedError(opt.calib_type)
    else:
        raise NotImplementedError(opt.exp_type)

    total_steps = 0
    eval_first_flg = True
    if opt.phase == 'test': # directly goes to test
        opt.epoch_count = 0
        opt.niter = 0
        opt.niter_decay = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        if epoch >= 1:
            eval_first_flg = False

        if opt.phase == 'train' and ((opt.niter + opt.niter_decay + 1 - epoch) < 2): # automatically saving the last result
            _config["save_prediction"] = True

        if epoch < opt.pretrain_epoch:
            train_loader = pretrain_loader
        else:
            train_loader = aleatrain_loader

        epoch_start_time = time.time()
        iter_data_time = time.time()
        np.random.seed()
        if opt.phase == 'train' and epoch != opt.epoch_count:
            for i, train_batch in tqdm(enumerate(train_loader), total = train_loader.dataset.size // train_loader.batch_size - 1):

                ## prepare data
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += train_loader.batch_size

                ## avoid incorrect batch size
                if train_batch["img"].shape[0] != train_loader.batch_size:
                    continue

                train_input = {'img': train_batch["img"],
                               'lb': train_batch["lb"]}

                ## run a training step
                if epoch < opt.pretrain_epoch:
                    model.set_input(train_input)
                else:
                    model.set_input_aug_sup(train_input)

                if epoch >= opt.pretrain_epoch:
                    model.optimize_parameters( is_pretrain = False, flip_mode = opt.dae_flip_mode  )
                else:
                    model.optimize_parameters( is_pretrain = True , flip_mode = opt.dae_flip_mode )

                ## display training losses
                if total_steps % opt.display_freq == 0:
                    tr_viz = model.get_current_visuals_tr()
                    model.plot_image_in_tb(tb_writer, tr_viz)

                if total_steps % opt.print_freq == 0:
                    tr_error = model.get_current_errors_tr()
                    t = (time.time() - iter_start_time) / train_loader.batch_size
                    model.track_scalar_in_tb(tb_writer, tr_error, total_steps)

                ## run and display validation losses
                if total_steps % opt.validation_freq == 0:
                    with torch.no_grad():
                        try:
                            val_batch = next(val_loader) # FIXME: use a nicer way. This is too ugly
                        except:
                            val_loader = iter(DataLoader(dataset = val_source_set, num_workers = opt.nThreads,\
                                batch_size = 1, drop_last = True, shuffle = True))
                            val_batch = next(val_loader)


                        val_input = {
                                'img': val_batch["img"],
                                'lb':  val_batch["lb"]
                                }
                        model.set_input(val_input)
                        model.validate()
                        val_errors = model.get_current_errors_val()

                    if total_steps % opt.display_freq == 0:
                        val_viz = model.get_current_visuals_val()
                        model.plot_image_in_tb(tb_writer, val_viz, total_steps)

                        val_errors = model.get_current_errors_val()
                        model.track_scalar_in_tb(tb_writer, val_errors, total_steps)

                iter_data_time = time.time()

        if ((epoch % opt.infer_epoch_freq == 0) and (epoch >= opt.pretrain_epoch)) or (eval_first_flg == True):
            eval_first_flg = False
            t0  = time.time()
            print('infering the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            with torch.no_grad():
                print(f'Starting inferring on the OOD data, for the ease of observation... ')
                torch.cuda.empty_cache()
                preds, error_table, error_dict, domain_list = prediction_wrapper(model, test_ood_loader, opt, epoch, label_name, fold = opt.eval_fold)

                for _metric, _table in error_table.items():

                    _run.log_scalar(f'raw{_metric}', _table.tolist())
                    _run.log_scalar(f'mean{_metric}', error_dict[_metric]['overall'] )

                    for _dm in domain_list:
                        _run.log_scalar(f'mean{_metric}_{_dm}', error_dict[_metric][f'domain_{_dm}_overall'])
                        _run.log_scalar(f'raw{_metric}_{_dm}', error_dict[_metric][f'domain_{_dm}_table'].tolist())

                if (opt.phase != 'test') and (_config['save_prediction'] is False):
                    del preds
                    torch.cuda.empty_cache()
                print(f'Start inferring on clean data... ')
                src_preds, error_table, error_dict, _ = prediction_wrapper(model, test_source_loader, opt, epoch, label_name, fold = opt.eval_fold)

                for _metric, _table in error_table.items():
                    _run.log_scalar(f'source_raw{_metric}', _table.tolist())
                    _run.log_scalar(f'source_mean{_metric}', error_dict[_metric]['overall'] )

                if _config["save_prediction"]:
                    save_which = ['pred']
                    save_which.append('logits')
                    save_which.append('temp_map')

                    ###### prediction
                    for _savekey in save_which:
                        for jj, _preds in enumerate([preds, src_preds]):
                            if jj == 0:
                                save_set = test_ood_set
                            elif jj == 1:
                                save_set = test_source_set

                            for scan_id, comp in _preds.items():
                                _pred = comp[_savekey]

                                if opt.data_name == 'MSC':
                                    scan_id, card_frame = scan_id.split("_")

                                    itk_pred = sitk.GetImageFromArray(_pred.cpu().numpy())
                                    itk_pred.SetSpacing(  save_set.info_by_scan[scan_id][card_frame]["spacing"] )
                                    itk_pred.SetOrigin(   save_set.info_by_scan[scan_id][card_frame]["origin"] )
                                    itk_pred.SetDirection(save_set.info_by_scan[scan_id][card_frame]["direction"] )

                                else:
                                    itk_pred = sitk.GetImageFromArray(_pred.cpu().numpy())
                                    itk_pred.SetSpacing(  save_set.info_by_scan[scan_id]["spacing"] )
                                    itk_pred.SetOrigin(   save_set.info_by_scan[scan_id]["origin"] )
                                    itk_pred.SetDirection(save_set.info_by_scan[scan_id]["direction"] )
                                    card_frame = ""

                                fid = os.path.join(model.pred_dir, f'{_savekey}_{scan_id}_frame_{card_frame}_epoch_{epoch}.nii.gz')
                                sitk.WriteImage(itk_pred, fid, True)
                                _log.info(f'###### {fid} has been saved ######')

                t1 = time.time()
                #if opt.tr_domain == 'C': # just in case of memory issue with domain prostate-C
                del src_preds
                torch.cuda.empty_cache()
                print("End of model inference, which takes {} seconds".format(t1 - t0))

        if opt.phase == 'test':
            return

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch >= opt.pretrain_epoch:
            model.update_learning_rate()

        if (opt.early_stop_epoch > 0) and (epoch >= opt.early_stop_epoch):
            return
