SCRIPT=tune_proposed.py
GPUID1=$1
NOTE=$2
NUM_WORKER=4 # 0 for debugging. It should be 4
MODEL='scale_unet'
CPT='test_proposed'

PRINT_FREQ=100000
VAL_FREQ=200000
TEST_EPOCH=400
EXP_TYPE='tune_calibrator'
CALIB_TYPE='proposed'
ALEA_N_REP=60 # for inference

BSIZE=1
BETA1=0.99

LAMBDA_WCE=0.0
LAMBDA_DICE=0.0
LAMBDA_CALIB=1.0

SAVE_EPOCH=400
SAVE_PRED=False

# restore model
DATASET='ACDC'
CHECKPOINTS_DIR="./exp_logs/TEST_CALIB_${DATASET}"
NITER=0
NITER_DECAY=1600 # total epoches
NITER_PRETRAIN=800 # epoches for training the dae
IMG_SIZE=192

OPTM_TYPE='adam'
LR=0.001
DAE_LR=0.001
ADAM_L2=0.00003

ALL_CVS=( 0 1 2 ) 
ALL_RUNS=( 1 2 3 ) # repeat the experiments for each 
UNET_SCALE_BASE=32
UNET_USE_MC=False

# validation fold
NCLASS=4

# label blender
LR_POLICY='step'
LR_DECAY_ITERS=200

# DAE options
DAE_DROP_P=0.5 # dropout rate for DAE encoder. More aggresive dropout makes the prior stronger
DAE_SCALE_BASE=8 # size of DAE. Changing to 16 may make it stonger
LAMBDA_DAE=1.0
DAE_MODE=unet
DAE_BILIN=False
FLIP_MODE=True
DAE_BATCHSIZE=20

for EVAL_FOLD in "${ALL_CVS[@]}"
do
    for RUN in "${ALL_RUNS[@]}"
    do
    RELOAD_MODEL="./frozen_segmenters/net_Seg_ev${EVAL_FOLD}.pth"
    RELOAD_DAE="./exp_logs/CALIB_ACDC/Calibration_tune_proposed_tune_calibrator_proposed_ev${EVAL_FOLD}/${RUN}/snapshots/1600_net_DAE.pth"
    RELOAD_CALIB="./exp_logs/CALIB_ACDC/Calibration_tune_proposed_tune_calibrator_proposed_ev${EVAL_FOLD}/${RUN}/snapshots/1600_net_Calib.pth"
    set -ex
    export CUDA_VISIBLE_DEVICES=$GPUID1

    NAME=${NOTE}_${CPT}_${EXP_TYPE}_${CALIB_TYPE}_ev${EVAL_FOLD}_rep${ALEA_N_REP}
    LOAD_DIR=$NAME
    python3 $SCRIPT with exp_type=$EXP_TYPE \
        name=$NAME \
        model=$MODEL \
        nThreads=$NUM_WORKER \
        print_freq=$PRINT_FREQ \
        validation_freq=$VAL_FREQ \
        batchSize=$BSIZE \
        lambda_wce=$LAMBDA_WCE \
        lambda_dice=$LAMBDA_DICE \
        save_epoch_freq=$SAVE_EPOCH \
        load_dir=$LOAD_DIR \
        checkpoints_dir=$CHECKPOINTS_DIR \
        which_epoch=$WHICH_EPOCH \
        infer_epoch_freq=$TEST_EPOCH \
        niter=$NITER \
        niter_decay=$NITER_DECAY \
        fineSize=$IMG_SIZE \
        lr=$LR \
        adam_weight_decay=$ADAM_L2 \
        data_name=$DATASET \
        eval_fold=$EVAL_FOLD \
        nclass=$NCLASS \
        optimizer=$OPTM_TYPE \
        save_prediction=$SAVE_PRED \
        display_freq=$PRINT_FREQ \
        unet_scale_base=$UNET_SCALE_BASE \
        reload_model_fid=$RELOAD_MODEL \
        reload_calib_fid=$RELOAD_CALIB \
        reload_dae_fid=$RELOAD_DAE \
        phase='test' \
        continue_train=True \
        lambda_calib=$LAMBDA_CALIB \
        calib_type=$CALIB_TYPE \
        beta1=$BETA1 \
        lr_decay_iters=$LR_DECAY_ITERS \
        lr_policy=$LR_POLICY \
        acdcc_domains='["Bias", "Motion", "Ghosting", "Spike"]' \
        dae_enc_drop_p=$DAE_DROP_P \
        dae_scale_base=$DAE_SCALE_BASE \
        lambda_dae=$LAMBDA_DAE \
        dae_lr=$DAE_LR \
        dae_mode=$DAE_MODE \
        dae_bilinear=$DAE_BILIN \
        pretrain_epoch=$NITER_PRETRAIN \
        alea_n_rep=$ALEA_N_REP \
        dae_flip_mode=$FLIP_MODE \
        daebatchSize=$DAE_BATCHSIZE
    done
done

