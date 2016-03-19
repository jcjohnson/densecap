local M = {}

function M.setup(opt)

   local model
   if opt.checkpoint_start_from == '' then
     print('initializing a DenseCap model from scratch...')
     model = DenseCapModel(opt)
   else
     print('initializing a DenseCap model from ' .. opt.checkpoint_start_from)
     model = torch.load(opt.checkpoint_start_from).model  
     model.opt.objectness_weight = opt.objectness_weight
     model.nets.detection_module.opt.obj_weight = opt.objectness_weight
     model.opt.box_reg_weight = opt.box_reg_weight
     model.nets.box_reg_crit.w = opt.final_box_reg_weight
     model.opt.classification_weight = opt.classification_weight
     local rpn = model.nets.detection_module.nets.rpn
     rpn:findModules('nn.RegularizeLayer')[1].w = opt.box_reg_decay
     model.opt.sampler_high_thresh = opt.iou_high_thresh
     model.opt.sampler_low_thresh = opt.iou_low_thresh
     model.opt.train_remove_outbounds_boxes = opt.train_remove_outbounds_boxes
     model.opt.captioning_weight = opt.captioning_weight
   end

   -- Find all Dropout layers and set their probabilities according to provided option
   local dropout_modules = model.nets.recog_base:findModules('nn.Dropout')
   for i, dropout_module in ipairs(dropout_modules) do
     dropout_module.p = opt.drop_prob
   end

   return model
end

return M