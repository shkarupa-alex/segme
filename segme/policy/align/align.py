from segme.policy.align.deconv import DeconvolutionFeatureAlignment
from segme.policy.align.deform import DeformableFeatureAlignment
from segme.policy.align.fade import FadeFeatureAlignment
from segme.policy.align.impf import ImplicitFeatureAlignment
from segme.policy.align.linear import BilinearFeatureAlignment
from segme.policy.align.sapa import SapaFeatureAlignment
from segme.policy.registry import LayerRegistry

ALIGNERS = LayerRegistry()
ALIGNERS.register('deconv3')(DeconvolutionFeatureAlignment)
ALIGNERS.register('deconv4')({
    'class_name': 'SegMe>Policy>Align>DeconvolutionFeatureAlignment', 'config': {'kernel_size': 4}})
ALIGNERS.register('deform')(DeformableFeatureAlignment)
ALIGNERS.register('fade')(FadeFeatureAlignment)
ALIGNERS.register('impf')(ImplicitFeatureAlignment)
ALIGNERS.register('linear')(BilinearFeatureAlignment)
ALIGNERS.register('sapa')(SapaFeatureAlignment)
