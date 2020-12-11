import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from ..sample import ClassificationUncertainty, classification_uncertainty
from ..sample import PointSample, point_sample
from ..sample import UncertainPointsWithRandomness, uncertain_points_with_randomness
from ..sample import UncertainPointsCoordsOnGrid, uncertain_points_coords_on_grid
from ....testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class TestClassificationUncertainty(keras_parameterized.TestCase):
    def setUp(self):
        super(TestClassificationUncertainty, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestClassificationUncertainty, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            ClassificationUncertainty,
            kwargs={},
            input_shape=[2, 16, 16, 10],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ClassificationUncertainty,
            kwargs={'from_logits': True},
            input_shape=[2, 16, 16, 10],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            ClassificationUncertainty,
            kwargs={'from_logits': True},
            input_shape=[2, 16, 16, 1],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16],
            expected_output_dtype='float32'
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        testing_utils.layer_test(
            ClassificationUncertainty,
            kwargs={},
            input_shape=[2, 16, 16, 10],
            input_dtype='float16',
            expected_output_shape=[None, 16, 16],
            expected_output_dtype='float16'
        )
        testing_utils.layer_test(
            ClassificationUncertainty,
            kwargs={},
            input_shape=[2, 16, 16, 10],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16],
            expected_output_dtype='float32'
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_values_multiclass(self):
        logits = [
            [[[0.6233578610415071, 0.8056737505372528, 0.5903172324801726, 0.5989098301399469],
              [0.5953177670614429, 0.4163235781883602, 0.7641045609448427, 0.23088485030546668]],
             [[0.7229075921797835, 0.8142660844216991, 0.21169034003429144, 0.6943984544484763],
              [0.6482516524195218, 0.9682387016057932, 0.8892865070415478, 0.5482739379396733]]],
            [[[0.4122575728028818, 0.4408815231416934, 0.13553924712504417, 0.42392864929129337],
              [0.2893530010764078, 0.13001759512913869, 0.6532980432279508, 0.829134938674454]],
             [[0.1822078508133197, 0.13021842311416976, 0.8015916731899908, 0.7391136434992175],
              [0.4451503035491462, 0.19348414428342098, 0.7748658737495627, 0.2726703802421446]]],
            [[[0.7332050156771194, 0.36165601273459214, 0.845607403578425, 0.23717402559095113],
              [0.10935190459070654, 0.09170544068727104, 0.7964688253273546, 0.17034152753338339]],
             [[0.0554124105966084, 0.00029060428317029263, 0.47972018205193934, 0.31558638322957855],
              [0.7669434550898543, 0.03334837592418649, 0.1331010045532316, 0.649419055774553]]]
        ]
        expected = [
            [[-0.18231588949574562, -0.1687867938833998],
             [-0.09135849224191561, -0.0789521945642454]],
            [[-0.016952873850400008, -0.17583689544650316],
             [-0.06247802969077332, -0.32971557020041653]],
            [[-0.11240238790130563, -0.6261272977939712],
             [-0.1641337988223608, -0.11752439931530134]]
        ]
        result = classification_uncertainty(tf.convert_to_tensor(logits), from_logits=False)
        self.assertAllClose(expected, result)

    def test_values_binary(self):
        logits = [
            [[[0.6233578610415071],
              [0.5953177670614429]],
             [[0.7229075921797835],
              [0.6482516524195218]]],
            [[[0.4122575728028818],
              [0.2893530010764078]],
             [[0.1822078508133197],
              [0.4451503035491462]]],
            [[[0.7332050156771194],
              [0.1093519045907065]],
             [[0.0554124105966084],
              [0.7669434550898543]]]
        ]
        expected = [
            [[-0.24671566486358643, -0.1906355619430542],
             [-0.44581520557403564, -0.29650330543518066]],
            [[-0.17548486590385437, -0.4212939739227295],
             [-0.6355843544006348, -0.10969936847686768]],
            [[-0.4664100408554077, -0.7812962532043457],
             [-0.8891751766204834, -0.5338869094848633]]]
        result = classification_uncertainty(tf.convert_to_tensor(logits), from_logits=False)
        self.assertAllClose(expected, result)


@keras_parameterized.run_all_keras_modes
class TestPointSample(keras_parameterized.TestCase):
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
        [[0.02364788512753946, 0.35329979440315273], [0.479255256132052, 0.6999485200720145],
         [0.3900163005799949, 0.022680914068735847]],
        [[0.8125130523743317, 0.8080015683668195], [0.779621045944294, 0.3385469444846084],
         [0.28337906040280836, 0.8989370027436526]]
    ]
    grid_corner = [  # 2 x 3 x 2
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    ]
    bilinear_random = [
        [[0.4946522116661072, 0.4969025254249573], [0.5992345809936523, 0.38736796379089355],
         [0.02763419784605503, 0.3467414975166321]],
        [[0.6845056414604187, 0.12966862320899963], [0.5646132230758667, 0.40902620553970337],
         [0.1310044378042221, 0.3320242762565613]]]
    bilinear_corner = [
        [[0.1450890153646469, 0.24396833777427673], [0.24609315395355225, 0.2453467845916748],
         [0.026439279317855835, 0.24564866721630096]],
        [[0.01776115782558918, 0.056393008679151535], [0.13448593020439148, 0.015676435083150864],
         [0.16188979148864746, 0.12015028297901154]]]
    nearest_random = [
        [[0.82572340965271, 0.5206645727157593], [0.43934887647628784, 0.4548532962799072],
         [0.05154120549559593, 0.5912343263626099]],
        [[0.7110949158668518, 0.13519690930843353], [0.3620241582393646, 0.062319450080394745],
         [0.09822063893079758, 0.4689144492149353]]
    ]
    nearest_corner = [
        [[0.5803560614585876, 0.9758733510971069], [0.984372615814209, 0.9813871383666992], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.6475591659545898, 0.48060113191604614]]
    ]

    def setUp(self):
        super(TestPointSample, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestPointSample, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            PointSample,
            kwargs={'mode': 'bilinear'},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 20, 2)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 20, 10)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            PointSample,
            kwargs={'mode': 'nearest'},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 20, 2)],
            input_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 20, 10)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            PointSample,
            kwargs={'mode': 'bilinear'},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 20, 2)],
            input_dtypes=['int32', 'float32'],
            expected_output_shapes=[(None, 20, 10)],
            expected_output_dtypes=['int32']
        )
        layer_multi_io_test(
            PointSample,
            kwargs={'mode': 'nearest'},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 20, 2)],
            input_dtypes=['int32', 'float32'],
            expected_output_shapes=[(None, 20, 10)],
            expected_output_dtypes=['int32']
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            PointSample,
            kwargs={'mode': 'bilinear'},
            input_datas=[
                np.random.rand(2, 16, 16, 10).astype(np.float16),
                np.random.rand(2, 20, 2).astype(np.float16)],
            input_dtypes=['float16', 'float16'],
            expected_output_shapes=[(None, 20, 10)],
            expected_output_dtypes=['float16']
        )
        layer_multi_io_test(
            PointSample,
            kwargs={'mode': 'bilinear'},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 20, 2)],
            input_dtypes=['float32', 'float16'],
            expected_output_shapes=[(None, 20, 10)],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            PointSample,
            kwargs={'mode': 'bilinear'},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 20, 2)],
            input_dtypes=['int32', 'float16'],
            expected_output_shapes=[(None, 20, 10)],
            expected_output_dtypes=['int32']
        )
        layer_multi_io_test(
            PointSample,
            kwargs={'mode': 'bilinear'},
            input_datas=[np.random.rand(2, 16, 16, 10), np.random.rand(2, 20, 2)],
            input_dtypes=['int32', 'float32'],
            expected_output_shapes=[(None, 20, 10)],
            expected_output_dtypes=['int32']
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_values_bilinear_random(self):
        result = point_sample([
            tf.convert_to_tensor(self.features, 'float32'),
            tf.convert_to_tensor(self.grid_random, 'float32')])
        self.assertAllClose(self.bilinear_random, result)

    def test_values_bilinear_corner(self):
        result = point_sample([
            tf.convert_to_tensor(self.features, 'float32'),
            tf.convert_to_tensor(self.grid_corner, 'float32')])
        self.assertAllClose(self.bilinear_corner, result)

    def test_values_nearest_random(self):
        result = point_sample([
            tf.convert_to_tensor(self.features, 'float32'),
            tf.convert_to_tensor(self.grid_random, 'float32')], mode='nearest')
        self.assertAllClose(self.nearest_random, result)

    def test_values_nearest_corner(self):
        result = point_sample([
            tf.convert_to_tensor(self.features, 'float32'),
            tf.convert_to_tensor(self.grid_corner, 'float32')], mode='nearest')
        self.assertAllClose(self.nearest_corner, result)


@keras_parameterized.run_all_keras_modes
class TestUncertainPointsWithRandomness(keras_parameterized.TestCase):
    def setUp(self):
        super(TestUncertainPointsWithRandomness, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestUncertainPointsWithRandomness, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            UncertainPointsWithRandomness,
            kwargs={'points': 5},
            input_shape=[2, 16, 16, 10],
            input_dtype='float32',
            expected_output_shape=[None, 5, 2],
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            UncertainPointsWithRandomness,
            kwargs={'points': 256},
            input_shape=[2, 16, 16, 10],
            input_dtype='float32',
            expected_output_shape=[None, 256, 2],
            expected_output_dtype='float32'
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        testing_utils.layer_test(
            UncertainPointsWithRandomness,
            kwargs={'points': 5},
            input_shape=[2, 16, 16, 10],
            input_dtype='float16',
            expected_output_shape=[None, 5, 2],
            expected_output_dtype='float16'
        )
        testing_utils.layer_test(
            UncertainPointsWithRandomness,
            kwargs={'points': 5},
            input_shape=[2, 16, 16, 10],
            input_dtype='float32',
            expected_output_shape=[None, 5, 2],
            expected_output_dtype='float32'
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_shorthand(self):
        uncertain_points_with_randomness(
            tf.convert_to_tensor(np.random.rand(2, 16, 16, 10).astype(np.float32)),
            points=3
        )


@keras_parameterized.run_all_keras_modes
class TestUncertainPointsCoordsOnGrid(keras_parameterized.TestCase):
    def setUp(self):
        super(TestUncertainPointsCoordsOnGrid, self).setUp()
        self.default_policy = tf.keras.mixed_precision.experimental.global_policy()

    def tearDown(self):
        super(TestUncertainPointsCoordsOnGrid, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy(self.default_policy)

    def test_layer(self):
        layer_multi_io_test(
            UncertainPointsCoordsOnGrid,
            kwargs={'points': 4},
            input_shapes=[(2, 16, 16, 2)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, None), (None, None, 2)],
            expected_output_dtypes=['int32', 'float32']
        )
        layer_multi_io_test(
            UncertainPointsCoordsOnGrid,
            kwargs={'points': 128},
            input_shapes=[(2, 4, 4, 3)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, None), (None, None, 2)],
            expected_output_dtypes=['int32', 'float32']
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
        layer_multi_io_test(
            UncertainPointsCoordsOnGrid,
            kwargs={'points': 4},
            input_shapes=[(2, 16, 16, 4)],
            input_dtypes=['float16'],
            expected_output_shapes=[(None, None), (None, None, 2)],
            expected_output_dtypes=['int32', 'float16']
        )
        layer_multi_io_test(
            UncertainPointsCoordsOnGrid,
            kwargs={'points': 4},
            input_shapes=[(2, 16, 16, 4)],
            input_dtypes=['float32'],
            expected_output_shapes=[(None, None), (None, None, 2)],
            expected_output_dtypes=['int32', 'float32']
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_shapes_normal(self):
        layer = UncertainPointsCoordsOnGrid(points=3)

        shape0, shape1 = layer.compute_output_shape(tf.TensorShape((2, 16, 16, 2)))
        self.assertListEqual([2, 3], shape0.as_list())
        self.assertListEqual([2, 3, 2], shape1.as_list())

        shape0, shape1 = layer.compute_output_shape(tf.TensorShape((2, None, None)))
        self.assertListEqual([2, None], shape0.as_list())
        self.assertListEqual([2, None, 2], shape1.as_list())

    def test_shapes_overflow(self):
        layer = UncertainPointsCoordsOnGrid(points=128)

        shape0, shape1 = layer.compute_output_shape(tf.TensorShape((2, 4, 4, 3)))
        self.assertListEqual([2, 16], shape0.as_list())
        self.assertListEqual([2, 16, 2], shape1.as_list())

        shape0, shape1 = layer.compute_output_shape(tf.TensorShape((2, None, None)))
        self.assertListEqual([2, None], shape0.as_list())
        self.assertListEqual([2, None, 2], shape1.as_list())

    def test_values(self):
        features = [
            [[[-4.646135547873334, 4.646135547873341], [-1.375307800955708, 1.3753078009557083],
              [-0.19970314154199903, 0.19970314154199914], [-4.3998986936639355, 4.3998986936639355],
              [-0.012143553949694073, 0.012143553949693964], [-2.812052515006374, 2.812052515006374],
              [-0.28009143447790935, 0.2800914344779095], [-0.3410848538738984, 0.3410848538738984]],
             [[-0.8154843632649794, 0.8154843632649794], [-2.4569516435868444, 2.4569516435868444],
              [-4.533784815637301, 4.533784815637301], [-1.3323587211882135, 1.3323587211882133],
              [-0.5381762159142277, 0.5381762159142279], [-0.7075404780740958, 0.7075404780740959],
              [-0.2146819039547535, 0.21468190395475326], [-0.18520836254753287, 0.18520836254753298]],
             [[-1.4929532065913944, 1.4929532065913942], [-1.2350237763328111, 1.235023776332811],
              [-0.35276945536068804, 0.35276945536068777], [-2.138654767051932, 2.138654767051932],
              [-1.963596407285422, 1.963596407285422], [-1.5506836267571715, 1.5506836267571715],
              [-2.850067419708923, 2.8500674197089233], [-0.8353743119478461, 0.835374311947846]]],
            [[[-1.2828382489979584, 1.2828382489979582], [-0.36070468796730715, 0.3607046879673072],
              [-3.5566671019296385, 3.5566671019296368], [-0.15716493097401, 0.15716493097401],
              [-0.6395075823993049, 0.6395075823993049], [-1.190738402836908, 1.190738402836908],
              [-3.63854207446555, 3.638542074465548], [-0.6774810571423963, 0.6774810571423961]],
             [[-0.935294875219582, 0.9352948752195819], [-1.574082064124059, 1.574082064124059],
              [-1.8953593847756893, 1.8953593847756893], [-2.858366955207996, 2.8583669552079947],
              [-0.3589678358404056, 0.35896783584040537], [-4.351585627657827, 4.351585627657823],
              [-5.684673329656756, 5.684673329656773], [-3.169891776368414, 3.1698917763684156]],
             [[-0.8661320577270409, 0.8661320577270412], [-1.4512941726983248, 1.4512941726983246],
              [-0.387325148173885, 0.38732514817388514], [-0.23048132645067856, 0.23048132645067854],
              [-0.12871999764606243, 0.1287199976460623], [-0.918591007576712, 0.9185910075767121],
              [-1.4405774078428244, 1.440577407842824], [-2.5533628319715125, 2.5533628319715134]]]
        ]
        indices = [[4, 15, 2], [20, 3, 19]]
        coords = [
            [[0.5625, 0.1666666716337204], [0.9375, 0.5], [0.3125, 0.1666666716337204]],
            [[0.5625, 0.8333333730697632], [0.4375, 0.1666666716337204], [0.4375, 0.8333333730697632]]
        ]
        result0, result1 = uncertain_points_coords_on_grid(tf.convert_to_tensor(features), points=3)
        self.assertAllClose(indices, result0)
        self.assertAllClose(coords, result1)


if __name__ == '__main__':
    tf.test.main()
