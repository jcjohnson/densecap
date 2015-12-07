require 'torch'

print 'got here'

densecap = {}

torch.include('densecap', 'densecap.ApplyBoxTransform')

return densecap
