import numpy as np
import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from ..gridsample import GridSample, grid_sample
from ...testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class TestGridSample(test_combinations.TestCase):
    features = [  # 2 x 3 x 8 x 2
        [[[0.5803560530488882, 0.9758733582668992], [0.3060797016509539, 0.8958330296139824],
          [0.04392514449205642, 0.6417255796191343], [0.05154120577339916, 0.5912343257885164],
          [0.4269209266455476, 0.47205654127613983], [0.4866003048338161, 0.8279622963285193],
          [0.6993539913503232, 0.37485069691849404], [0.10575711773669993, 0.9825946582539481]],
         [[0.8257234224540615, 0.5206645988316856], [0.7093410631788002, 0.1437249653745346],
          [0.17192861245295066, 0.7740607221316901], [0.8234077712492931, 0.3505931790314334],
          [0.8162857557989622, 0.27297732800530183], [0.9250948423285806, 0.29757936684731723],
          [0.6986999087019943, 0.3079055707545305], [0.21179192038896555, 0.3171553461301925]],
         [[0.9843725970754822, 0.9813871174443999], [0.44842257348612125, 0.130380716609253],
          [0.7722375422750853, 0.10364272271858943], [0.4393488700955268, 0.4548532986245398],
          [0.47505635832467275, 0.37804489426244103], [0.8258436454865106, 0.35831259847880437],
          [0.8239819761272512, 0.8828993842018817], [0.49231643099716704, 0.23657644268638445]]],
        [[[0.3470358737217457, 0.16735936338606427], [0.9517472467867465, 0.9799591409158379],
          [0.35939672961575697, 0.5674854226639825], [0.7903688431119552, 0.8681517646006246],
          [0.19527231727607397, 0.6249876602109309], [0.6353167539043052, 0.1460755017217389],
          [0.6647652592142212, 0.8747084010404066], [0.5379436958925545, 0.06270574106266635]],
         [[0.05861333972611882, 0.532425985136336], [0.36967274900613223, 0.018303947235997597],
          [0.7088710957558425, 0.01017874807266328], [0.6841144173469206, 0.14115792582714903],
          [0.2526197790318786, 0.30531102977329816], [0.8022331882515179, 0.4018300665076793],
          [0.36202417007484644, 0.06231945138957096], [0.3911262625629873, 0.23054775030088803]],
         [[0.6475591413630237, 0.4806011315711167], [0.37673455563997893, 0.2305467881152683],
          [0.09822063759763167, 0.4689144452408287], [0.05816316129992405, 0.5752466820324968],
          [0.4698720844783306, 0.9101509631176331], [0.6000989729234223, 0.9142169214834011],
          [0.7110949212180566, 0.1351969100128616], [0.07104463082685841, 0.22557203748208543]]]
    ]
    grid_random = [  # 2 x 3 x 2
        [[[-0.9527042297449211, -0.29340041119369453], [-0.041489487735896, 0.39989704014402894]],
         [[-0.21996739884001015, -0.9546381718625283], [0.6250261047486634, 0.6160031367336389]]],
        [[[0.6250261047486634, 0.6160031367336389], [0.5592420918885881, -0.32290611103078315]],
         [[-0.4332418791943833, 0.7978740054873052], [-0.21996739884001015, -0.9546381718625283]]]]
    grid_corner = [  # 2 x 3 x 2
        [[[-1.0, -1.0], [-1.0, 1.0]], [[1.0, -1.0], [1.0, 1.0]]],
        [[[1.0, 1.0], [1.0, -1.0]], [[-1.0, 1.0], [-1.0, -1.0]]]
    ]

    def setUp(self):
        super(TestGridSample, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(TestGridSample, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            GridSample,
            kwargs={'mode': 'bilinear', 'align_corners': False},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 4, 5, 2)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 4, 5, 10)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            GridSample,
            kwargs={'mode': 'nearest', 'align_corners': True},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 4, 5, 2)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 4, 5, 10)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            GridSample,
            kwargs={'mode': 'bilinear', 'align_corners': False},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 4, 5, 2)],
            input_dtypes=['int32', 'float32'],
            expected_output_shapes=[(None, 4, 5, 10)],
            expected_output_dtypes=['int32']
        )
        layer_multi_io_test(
            GridSample,
            kwargs={'mode': 'nearest', 'align_corners': True},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 4, 5, 2)],
            input_dtypes=['int32', 'float32'],
            expected_output_shapes=[(None, 4, 5, 10)],
            expected_output_dtypes=['int32']
        )

        mixed_precision.set_global_policy('mixed_float16')
        layer_multi_io_test(
            GridSample,
            kwargs={'mode': 'bilinear', 'align_corners': False},
            input_datas=[
                np.random.rand(2, 16, 16, 10).astype(np.float16),
                np.random.rand(2, 4, 5, 2).astype(np.float16)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 4, 5, 10)],
            expected_output_dtypes=['float16']
        )
        layer_multi_io_test(
            GridSample,
            kwargs={'mode': 'bilinear', 'align_corners': True},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 4, 5, 2)],
            input_dtypes=['float32', 'float16'],
            expected_output_shapes=[(None, 4, 5, 10)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            GridSample,
            kwargs={'mode': 'bilinear', 'align_corners': False},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 4, 5, 2)],
            input_dtypes=['int32', 'float16'],
            expected_output_shapes=[(None, 4, 5, 10)],
            expected_output_dtypes=['int32']
        )
        layer_multi_io_test(
            GridSample,
            kwargs={'mode': 'bilinear', 'align_corners': True},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 4, 5, 2)],
            input_dtypes=['int32', 'float32'],
            expected_output_shapes=[(None, 4, 5, 10)],
            expected_output_dtypes=['int32']
        )

    def test_values_bilinear_random(self):
        expected = [
            [[[0.4946522116661072, 0.4969025254249573], [0.5992345809936523, 0.38736796379089355]],
             [[0.02763419784605503, 0.3467414975166321], [0.8144252896308899, 0.8391402959823608]]],
            [[[0.6845056414604187, 0.12966862320899963], [0.5646132230758667, 0.40902620553970337]],
             [[0.1310044378042221, 0.3320242762565613], [0.3559671640396118, 0.4282688796520233]]]]
        result = grid_sample([
            np.array(self.features, 'float32'),
            np.array(self.grid_random, 'float32')], align_corners=False)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_values_bilinear_corner(self):
        expected = [
            [[[0.1450890153646469, 0.24396833777427673], [0.24609315395355225, 0.2453467845916748]],
             [[0.026439279317855835, 0.24564866721630096], [0.12307910621166229, 0.05914410948753357]]],
            [[[0.01776115782558918, 0.056393008679151535], [0.13448593020439148, 0.015676435083150864]],
             [[0.16188979148864746, 0.12015028297901154], [0.08675897121429443, 0.0418398417532444]]]]
        result = grid_sample([
            np.array(self.features, 'float32'),
            np.array(self.grid_corner, 'float32')], align_corners=False)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_values_nearest_random(self):
        expected = [
            [[[0.82572340965271, 0.5206645727157593], [0.43934887647628784, 0.4548532962799072]],
             [[0.05154120549559593, 0.5912343263626099], [0.8239820003509521, 0.8828994035720825]]],
            [[[0.7110949158668518, 0.13519690930843353], [0.3620241582393646, 0.062319450080394745]],
             [[0.09822063893079758, 0.4689144492149353], [0.7903688549995422, 0.8681517839431763]]]]
        result = grid_sample([
            np.array(self.features, 'float32'),
            np.array(self.grid_random, 'float32')], mode='nearest', align_corners=False)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_values_nearest_corner(self):
        expected = [
            [[[0.5803560614585876, 0.9758733510971069], [0.984372615814209, 0.9813871383666992]],
             [[0.0, 0.0], [0.0, 0.0]]],
            [[[0.0, 0.0], [0.0, 0.0]],
             [[0.6475591659545898, 0.48060113191604614], [0.34703588485717773, 0.1673593670129776]]]]
        result = grid_sample([
            np.array(self.features, 'float32'),
            np.array(self.grid_corner, 'float32')], mode='nearest', align_corners=False)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_values_bilinear_random_align(self):
        expected = [
            [[[0.7267985939979553, 0.6062460541725159], [0.6733735203742981, 0.36486396193504333]],
             [[0.07661650329828262, 0.5985114574432373], [0.8033915162086487, 0.5599108934402466]]],
            [[[0.6085014343261719, 0.29785940051078796], [0.6163655519485474, 0.3217147886753082]],
             [[0.2241607904434204, 0.37310990691185], [0.6748148798942566, 0.7561057806015015]]]]
        result = grid_sample([
            np.array(self.features, 'float32'),
            np.array(self.grid_random, 'float32')], align_corners=True)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_values_bilinear_corner_align(self):
        expected = [
            [[[0.5803560614585876, 0.9758733510971069], [0.984372615814209, 0.9813871383666992]],
             [[0.10575711727142334, 0.9825946688652039], [0.49231642484664917, 0.23657643795013428]]],
            [[[0.07104463130235672, 0.22557203471660614], [0.5379437208175659, 0.06270574033260345]],
             [[0.6475591659545898, 0.48060113191604614], [0.34703588485717773, 0.1673593670129776]]]]
        result = grid_sample([
            np.array(self.features, 'float32'),
            np.array(self.grid_corner, 'float32')], align_corners=True)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_values_nearest_random_align(self):
        expected = [
            [[[0.82572340965271, 0.5206645727157593], [0.823407769203186, 0.3505931794643402]],
             [[0.05154120549559593, 0.5912343263626099], [0.8239820003509521, 0.8828994035720825]]],
            [[[0.7110949158668518, 0.13519690930843353], [0.8022331595420837, 0.4018300771713257]],
             [[0.09822063893079758, 0.4689144492149353], [0.7903688549995422, 0.8681517839431763]]]]
        result = grid_sample([
            np.array(self.features, 'float32'),
            np.array(self.grid_random, 'float32')], mode='nearest', align_corners=True)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)

    def test_values_nearest_corner_align(self):
        expected = [
            [[[0.5803560614585876, 0.9758733510971069], [0.984372615814209, 0.9813871383666992]],
             [[0.10575711727142334, 0.9825946688652039], [0.49231642484664917, 0.23657643795013428]]],
            [[[0.07104463130235672, 0.22557203471660614], [0.5379437208175659, 0.06270574033260345]],
             [[0.6475591659545898, 0.48060113191604614], [0.34703588485717773, 0.1673593670129776]]]]
        result = grid_sample([
            np.array(self.features, 'float32'),
            np.array(self.grid_corner, 'float32')], mode='nearest', align_corners=True)
        result = self.evaluate(result)
        self.assertAllClose(expected, result)


if __name__ == '__main__':
    tf.test.main()
