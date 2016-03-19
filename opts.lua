local M = { }

function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a DenseCap model.')
  cmd:text()
  cmd:text('Options')

  -- Core ConvNet settings
  cmd:option('-backend', 'cudnn', 'nn|cudnn')

  -- Model settings
  cmd:option('-rpn_hidden_dim',512,'hidden size in the rpnnet')
  cmd:option('-sampler_batch_size',256,'batch size to use in the box sampler')
  cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
  cmd:option('-input_encoding_size',512,'what is the encoding size of each token in the vocabulary? (chars or words)')
  cmd:option('-sampler_high_thresh', 0.7, 'predicted boxes with IoU more than this with a positive box are considered as positive')
  cmd:option('-sampler_low_thresh', 0.3, 'predicted boxes with IoU less than this with a positive box are considered as negative')
  cmd:option('-train_remove_outbounds_boxes', 1,' Whether to ignore out-of-bounds boxes for sampling at training time')

  -- Loss function weights
  cmd:option('-mid_box_reg_weight',0.05,'what importance to assign to regressing bounding boxes well in rpn?')
  cmd:option('-mid_objectness_weight', 0.1, 'what importance to assign to pos/neg objectness labels?')
  cmd:option('-end_box_reg_weight', 0.1, 'what importance to assign to final class-specific bounding box regression?')
  cmd:option('-end_objectness_weight',0.1,'what importance to assign to classifying the correct class?')
  cmd:option('-captioning_weight',1.0,'what importance to assign to captioning, if present?')
  cmd:option('-weight_decay', 1e-6, 'L2 weight decay penalty strength')
  cmd:option('-box_reg_decay', 5e-5, 'Strength of a pull that boxes experience towards their anchor, to prevent wild drifts')

  -- Data input settings
  cmd:option('-data_h5','data/VG-regions.h5','path to the h5file containing the preprocessed dataset (made in prepro.py)')
  cmd:option('-data_json','data/VG-regions-dicts.json','path to the json file containing additional info (made in prepro.py)')
  cmd:option('-h5_read_all',false,'read the whole h5 dataset to memory? COCO images take several tens of GB might not fit in your RAM, need partial reading.')
  cmd:option('-proposal_regions_h5','','override RPN boxes with boxes from this h5 file (empty = don\'t override)')
  cmd:option('-debug_max_train_images', -1, 'for debugging: Cap #train images at this value to check that we can overfit. (-1 = disable)')

  -- Optimization
  cmd:option('-learning_rate',4e-6,'learning rate to use')
  cmd:option('-optim_beta1',0.9,'beta1 for adam')
  cmd:option('-optim_beta2',0.999,'beta2 for adam')
  cmd:option('-optim_epsilon',1e-8,'epsilon for smoothing')
  cmd:option('-drop_prob', 0.5, 'Dropout strength throughout the model.')
  cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
  cmd:option('-checkpoint_start_from', '', 'load model from a given checkpoint instead of random initialization.')
  cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
  cmd:option('-val_images_use', 100, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')

  -- Model checkpointing
  cmd:option('-save_checkpoint_every', 1000, 'how often to save a model checkpoint?')
  cmd:option('-checkpoint_path', 'checkpoint.t7', 'the name of the checkpoint file to use')

  -- Visualization
  cmd:option('-progress_dump_every', 100, 'Every how many iterations do we write a progress report to vis/out ?. 0 = disable.')
  cmd:option('-losses_log_every', 10, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

  -- Misc
  cmd:option('-id', '', 'an id identifying this run/job. can be used in cross-validation.')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
  cmd:option('-timing', false, 'whether to time parts of the net')
  cmd:option('-dump_all_losses', 0)
  cmd:option('-clip_final_boxes', 1,
             'whether to clip final boxes to image boundary; probably set to 0 for dense captioning')
  cmd:option('-eval_first_iteration',0,'evaluate on first iteration? 1 = do, 0 = dont.')

  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M