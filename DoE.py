######################## Simulator Code ###############################
# Author: Yue Shi
# Email: yueshi@usc.edu
#######################################################################

from pyDOE import lhs
from scipy.stats.distributions import norm, uniform
from scipy.stats import ttest_ind, levene
from sklearn.tree import DecisionTreeClassifier
from simulator import Simulator, HyperParameter

from dotmap import DotMap
import numpy as np
from collections import defaultdict, deque, Sequence
import operator
import copy
import operator
import random
import uuid
import time

sc = None
rand_state = {}

class SubRegion:
    def __init__(self, hyperParamRange, samples):
        self.hyperParamRange = hyperParamRange # hyperParameterRange
        self.samples = Samples(samples) # Samples class
        self.id = uuid.uuid4()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

class HyperParameterRange(DotMap):
    def to_array(self):
        return [self[k] if isinstance(self[k], Sequence) else (self[k], self[k]) for k in self.get_param_names()]

    def from_array(self, values):
        return HyperParameterRange({ k: v[0] if v[0] == v[1] else v for k, v in zip(self.get_param_names(), values) })

    def get_param_names(self):
        return sorted(self.keys())

    def get_range_param_names(self):
        return sorted(k for k in self.keys() if isinstance(self[k], Sequence))

class Sample:
    def __init__(self, X, Y=None, P=None, phi=None, Z=None):
        self.X = X #hyperParameter # hyperParameter class
        self.Y = Y #[ ... ] # Y
        self.P = P
        self.phi = phi #0 or 1 # phi
        self.Z = Z #0 or 1
        self.id = uuid.uuid4()

    def copy(self):
        return Sample(X=self.X, Y=self.Y, P=self.P, phi=self.phi, Z=self.Z)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

class Samples(list):
    def replicate(self, times):
        self[:] = sum([list(self) for _ in range(times)], [])


class DoEOption:
    def __init__(self, representation='NN', nSamples=10, nRunPerSample=10, goal=0.05, goalOp=operator.le,
        samplingMethod='LHS', maxLevel=3, convHigh=0.99, convTruncation=0.5, keepRatio=0.8, partitionCount=None,
        minParamRange=1e-3, minParamGap=5e-2, min_samples_leaf=5, min_samples_leaf_2=2, min_impurity_split=1e-1, max_leaf_nodes=None):
        """
        Args:
            nSamples: number of samples generated for each region.
            samplingMethod: the option to generate samples.
            maxLevel: the maximum level to explore.
            convHigh: the threshold which already meets the goal, and could stop split
            convTruncation: the threshold which is far from expectation, which could stop split
            keepRatio: Percentage of low group
        """
        self.representation = representation
        self.nSamples = nSamples
        self.nRunPerSample = nRunPerSample
        self.goal = goal
        self.goalOp = goalOp
        self.samplingMethod = samplingMethod
        self.maxLevel = maxLevel
        self.convHigh = convHigh
        self.convTruncation = convTruncation
        self.keepRatio = keepRatio
        self.partitionCount = partitionCount
        self.minParamRange = minParamRange
        self.minParamGap = minParamGap

        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_2 = min_samples_leaf_2
        self.min_impurity_split = min_impurity_split
        self.max_leaf_nodes = max_leaf_nodes

class DoE:
    def __init__(self, env, option=None):
        self.env = env
        self.option = option or DoEOption()

    @staticmethod
    def generateLHS(hyperParamRange, nSamples):
        indexhash = {}  # for all hyperparameter
        lowBounds = []
        highBounds = []
        index = 0

        interestedHPlen = 0
        for key, eachHpRng in hyperParamRange.iteritems():
            if isinstance(eachHpRng, Sequence) and eachHpRng[1] > eachHpRng[0]:
                lowBounds.append(eachHpRng[0])
                highBounds.append(eachHpRng[1])
                indexhash[key] = index
                interestedHPlen += 1
                index += 1

        design = lhs(interestedHPlen, samples=nSamples)

        for i in xrange(interestedHPlen):
            design[:,i] = uniform(loc=lowBounds[i], scale=highBounds[i]-lowBounds[i]).ppf(design[:,i])

        design = np.array(design)

        for key, eachHpRng in hyperParamRange.iteritems():
            if not isinstance(eachHpRng, Sequence):
                design = np.concatenate((design, np.full((nSamples,1), eachHpRng)), axis=1)
                indexhash[key] = index
                index += 1
            elif eachHpRng[1] <= eachHpRng[0]:
                design = np.concatenate((design, np.full((nSamples,1), eachHpRng[0])), axis=1)
                indexhash[key] = index
                index += 1

        samples = Samples([])
        for point in design:
            hyperParam = HyperParameter({ k: point[indexhash[k]] for k in hyperParamRange.get_param_names() })
            samples.append(Sample(hyperParam, 0.0, 0.0, 0.0))
        return samples

    def generateX(self, hyperParamRange, nSamples):
        '''
        Generate a list of hyperParameters from the defined hyperParameterRange.
        User could choose options:   1.random 2. LHS 3.orthogonal.
        Args:
            hyperParamRange: HyperParameterRange instance, the range of all hyperParameters.
        Ouput:
            Samples: Samples class instance.  Samples.samples=[Sample1, Sample2, Sample3, ....], only is X element would be valid now.
        '''

        if self.option.samplingMethod == 'LHS':
            samples = self.generateLHS(hyperParamRange, nSamples)

        return samples

    def runSimulator(self, samples):
        '''
        Args:
            Samples: The generated Samples.
        Ouput:
            Samples: Samples instance with Y value filled.
        '''

        def f(sample):
            # random.setstate(rand_state['random'])
            # np.random.set_state(rand_state['np'])

            start = time.time()
            sampleResult = Simulator(self.env).getTrain(representation=self.option.representation, s0=self.env.s0, hyperParam=sample.X)
            sample.Y = self.env.evaluateEach(sampleResult)
            # print 'rho=%.16f, Y=%.16f, time=%fs\n' % (sample.X.rho, sample.Y, time.time()-start),
            return sample

        return Samples(map(f, samples))

    def calY(self, samples, baselineValues):
        def f(sample):
            start = time.time()
            dup_samples = Samples([sample.copy() for _ in range(self.option.nRunPerSample)])
            dup_samples = self.runSimulator(dup_samples)
            values = [s.Y for s in dup_samples]

            if baselineValues:
                t, p = ttest_ind(baselineValues, values, equal_var=False)
                w, lp = levene(baselineValues, values, center='median')
                sample.Y = sum(values) / len(values)
                sample.P = p
                print 'rho=%.16f, gamma=%.16f, Y=%.16f, p=%.16f, lp=%.16f, t=%.16f, w=%.16f, time=%fs\n' % (sample.X.rho, sample.X.gamma, sample.Y, sample.P, lp, t, w, time.time()-start),
            else:
                sample.Y = sample.P = values[0]
                print 'rho=%.16f, gamma=%.16f, Y=%.16f, time=%fs\n' % (sample.X.rho, sample.X.gamma, sample.Y, time.time()-start),

            return sample

        return Samples(map(f, samples))

    def calPhi(self, samples):
        '''
        Args:
            Samples: Samples object
            goal: The optimization goal
        Ouput:
            Samples: Samples object with valid phi values
        '''
        for sample in samples:
            if self.option.goalOp(sample.P, self.option.goal):
                sample.phi = 1.0

        return samples

    def calZ(self, samples):
        '''
        Args:
            samples: Samples object
        Output:
            samples: Samples object with valid Z values
        '''

        direction = -1
        if self.option.goalOp == operator.gt or self.option.goalOp == operator.ge:
            direction = 1

        samples.sort(key=lambda sample: (direction * sample.P, sample.X.rho))
        nHighGroup = len(samples) * (1. - self.option.keepRatio)
        lastHighValue = None

        for i, sample in enumerate(samples):
            if i < nHighGroup:
                sample.Z = 0.0
                lastHighValue = sample.P
            elif sample.Y == lastHighValue:
                sample.Z = 0.0
            else:
                sample.Z = 1.0

        return samples

    def genCART(self, X, Z, minSamplesLeaf):
        '''
        call the CART library to generate the tree model.
        Args:
            Samples: Samples instance
        Ouput: model: the cart model
        '''
        clf = DecisionTreeClassifier(
            min_samples_leaf=minSamplesLeaf,
            min_impurity_split=self.option.min_impurity_split,
            max_leaf_nodes=self.option.max_leaf_nodes)
        clf.fit(X, Z)
        return clf

    def getSubRegions(self, samples, hyperParamRange):
        '''
        From the CART model, get the subregions.
        Arg:
            model: The cart model
        Output:
            subRegions: list of SubRegion.[SubRegion1,SubRegion2,....]  from the current model.
        '''
        X = [sample.X.to_array() for sample in samples]
        Z = [sample.Z for sample in samples]

        def shouldMerge(r1, r2, minParamGap):
            for (x1, x2) in zip(r1.hyperParamRange.to_array(), r2.hyperParamRange.to_array()):
                if x1 > x2:
                    x1, x2 = x2, x1

                if x1 != x2 and (x2[0] - x1[1] >= minParamGap):
                    return False

            return True

        def mergeRegion(r1, r2):
            r1.hyperParamRange = hyperParamRange.from_array(map(lambda (x1, x2): [min(x1[0], x2[0]), max(x1[1], x2[1])], zip(r1.hyperParamRange.to_array(), r2.hyperParamRange.to_array())))
            r1.samples.extend(r2.samples)

        def getSubRegionsInternal(minSamplesLeaf, minParamGap=0):
            model = self.genCART(X, Z, minSamplesLeaf)

            children_left = model.tree_.children_left
            children_right = model.tree_.children_right
            feature = model.tree_.feature
            threshold = model.tree_.threshold

            bounds = defaultdict(list)
            queue = deque([(0, hyperParamRange.to_array())])

            while len(queue):
                cur_index, cur_bound = queue.popleft()
                left_child = children_left[cur_index]
                right_child = children_right[cur_index]

                if left_child == right_child: # leaf node
                    bounds[cur_index] = copy.deepcopy(cur_bound)
                    continue

                f = feature[cur_index]
                t = threshold[cur_index]

                # left
                left_bound = copy.deepcopy(cur_bound)
                left_bound[f][1] = t
                queue.append((left_child, left_bound))

                # right
                right_bound = copy.deepcopy(cur_bound)
                right_bound[f][0] = t
                queue.append((right_child, right_bound))


            node_ids = model.apply(X)
            predicts = model.predict(X)
            regionsDict = {}

            for i, z in enumerate(predicts):
                if z != 1: continue

                node_id = node_ids[i]

                if node_id not in regionsDict:
                    regionsDict[node_id] = SubRegion(hyperParamRange.from_array(bounds[node_id]), [])

                regionsDict[node_id].samples.append(samples[i])

            regions = regionsDict.values()
            regions.sort(key=lambda region: region.hyperParamRange.to_array())

            merged = []
            for region in regions:
                if len(merged) and shouldMerge(merged[-1], region, minParamGap):
                    mergeRegion(merged[-1], region)
                else:
                    merged.append(region)
            return merged

        regions = getSubRegionsInternal(self.option.min_samples_leaf, self.option.minParamGap)
        if len(regions) <= 1:
            regions = getSubRegionsInternal(self.option.min_samples_leaf_2, self.option.minParamGap)
        if len(regions) <= 1:
            regions = getSubRegionsInternal(self.option.min_samples_leaf)
        if len(regions) <= 1:
            regions = getSubRegionsInternal(self.option.min_samples_leaf_2)

        print 'regions:', len(regions)
        return regions

    @staticmethod
    def calBeta(subRegion):
        '''
        Calculate the beta value of the subRegion, #samples_with_phi=1 / #samples_in_subRegion
        Args:
            subRegion: subRegion instance.
        output:
            beta_subRegion: the beta value of the subregion.
        '''

        beta = 1. * sum(1 for sample in subRegion.samples if sample.phi == 1) / len(subRegion.samples)
        return beta

    def sequentialCART(self, initHyperParamRange, baselineHyperParamRange=None):

        '''
        The sequentialCART algorithm
        Args:
            initHyperParamRange: The inital hyperParameter range .
        Output:
            hyperParametersRanges: the list of regions that could meet our goal.
        '''

        # rand_state['random'] = random.getstate()
        # rand_state['np'] = np.random.get_state()

        option = self.option

        if baselineHyperParamRange:
            baselineSamples = self.generateX(baselineHyperParamRange, option.nRunPerSample)
            baselineSamples = self.runSimulator(baselineSamples)
            baselineValues = [sample.Y for sample in baselineSamples]
        else:
            baselineValues = None
            option.nRunPerSample = 1

        results = []

        initRegion = SubRegion(initHyperParamRange, [])
        parentQueue = [initRegion]

        for level in range(option.maxLevel):
            childQueue = []

            for subRegion in parentQueue:
                # random.setstate(rand_state['random'])
                # np.random.set_state(rand_state['np'])

                samples = self.generateX(subRegion.hyperParamRange, option.nSamples)
                samples = self.calY(samples, baselineValues)
                samples = self.calPhi(samples)
                samples = self.calZ(samples)

                # random.setstate(rand_state['random'])
                # np.random.set_state(rand_state['np'])

                subRegions = self.getSubRegions(samples, subRegion.hyperParamRange)

                for childRegion in subRegions:
                    beta = self.calBeta(childRegion)

                    print 'checking:', childRegion.hyperParamRange.rho, childRegion.hyperParamRange.gamma, 'beta:', beta, 'samples:', len(childRegion.samples)

                    if beta >= option.convHigh:
                        results.append(childRegion.hyperParamRange)
                        continue

                    if beta <= option.convTruncation:
                        continue

                    childQueue.append(childRegion)

            parentQueue = childQueue

        return results

    def sequentialCART2(self, initHyperParamRange, baselineHyperParamRange=None):
        '''
        The sequentialCART algorithm
        Args:
            initHyperParamRange: The inital hyperParameter range .
        Output:
            hyperParamRanges: the list of regions that could meet our goal.
        '''

        # rand_state['random'] = random.getstate()
        # rand_state['np'] = np.random.get_state()

        option = self.option

        rangeParamNames = initHyperParamRange.get_range_param_names()

        def getParamInfo(param):
            info = []

            for k in rangeParamNames:
                if isinstance(param[k], Sequence):
                    info.append('{}=[{:.16f},{:.16f}]'.format(k, param[k][0], param[k][1]))
                else:
                    info.append('{}={:.16f}'.format(k, param[k]))

            return ', '.join(info)


        def populateSamples(region):
            # random.setstate(rand_state['random'])
            # np.random.set_state(rand_state['np'])

            region.samples = self.generateX(region.hyperParamRange, option.nSamples)
            # region.samples.replicate(3)
            return region

        def sampleCalY(sample):
            # random.setstate(rand_state['random'])
            # np.random.set_state(rand_state['np'])

            start = time.time()
            sampleResult = Simulator(self.env).getTrain(representation=option.representation, s0=self.env.s0, hyperParam=sample.X)
            sample.Y = self.env.evaluateEach(sampleResult)
            print '%s, Y=%.16f, time=%fs\n' % (getParamInfo(sample.X), sample.Y, time.time()-start),
            return sample

        def regionCalPhi(region):
            region.samples = self.calPhi(region.samples)
            return region

        def regionCalZ(region):
            region.samples = self.calZ(region.samples)
            return region

        def regionCalBeta(region):
            region.beta = self.calBeta(region)
            print 'checking:', getParamInfo(region.hyperParamRange), 'beta:', region.beta, 'samples:', len(region.samples)
            return region

        def regionGetSubRegions(region):
            # random.setstate(rand_state['random'])
            # np.random.set_state(rand_state['np'])

            return self.getSubRegions(region.samples, region.hyperParamRange)

        def computeStat(sample, baselineValues, values):
            if baselineValues:
                t, p = ttest_ind(baselineValues, values, equal_var=False)
                w, lp = levene(baselineValues, values, center='median')
                sample.Y = sum(values) / len(values)
                sample.P = p
                print '%s, Y=%.16f, p=%.16f, lp=%.16f, t=%.16f, w=%.16f\n' % (getParamInfo(sample.X), sample.Y, sample.P, lp, t, w),
            else:
                sample.Y = sample.P = values[0]

            return sample

        def assignSamples(region, samples):
            region.samples = Samples(samples)
            return region

        def setRandomSeed(idx, it):
            seed = idx + random.randint(0, 2**31)
            random.seed(seed)
            np.random.seed(seed)
            return it

        def checkParamRange(region):
            if not option.minParamRange or option.minParamRange == -1:
                return True

            for k in rangeParamNames:
                if region.hyperParamRange[k][1] - region.hyperParamRange[k][0] >= option.minParamRange:
                    return True

            return False

        from pyspark import SparkContext, SparkConf

        global sc
        if not sc:
            sc = SparkContext(conf=SparkConf().setMaster('local[*]'))
            sc.setLogLevel('WARN')


        if baselineHyperParamRange:
            baselineSamples = self.generateX(baselineHyperParamRange, option.nRunPerSample)
            baselineValues = sc.parallelize(baselineSamples, option.partitionCount) \
                .mapPartitionsWithIndex(lambda idx, it: setRandomSeed(idx, it)) \
                .map(sampleCalY) \
                .map(lambda x: x.Y) \
                .collect()
        else:
            baselineValues = None
            option.nRunPerSample = 1

        initRegion = SubRegion(initHyperParamRange, [])
        regionsRDD = sc.parallelize([initRegion], option.partitionCount)
        partitionCount = regionsRDD.getNumPartitions()
        resultsRDD = None
        #.groupByKey() \   .map(lambda x: x[1]) and .mapValues(sampleCalY)
        #.map(lambda (region, sample): (region, sampleCalY(sample))) \ .mapValues(sampleCalY) and .mapValues(lambda x: [x])
        for level in range(option.maxLevel):
            regionsRDD = regionsRDD \
                .map(populateSamples) \
                .flatMap(lambda region: [((region, sample), sample.copy()) for sample in region.samples for _ in range(option.nRunPerSample)]) \
                .zipWithIndex() \
                .map(lambda x: x[::-1]) \
                .partitionBy(partitionCount, partitionFunc=lambda x: x % partitionCount) \
                .map(lambda x: x[1]) \
                .mapPartitionsWithIndex(lambda idx, it: setRandomSeed(idx, it)) \
                .mapValues(sampleCalY) \
                .mapValues(lambda x: [x]) \
                .reduceByKey(lambda a, b: a + b) \
                .map(lambda ((region, sample), samples): (region, computeStat(sample, baselineValues, [s.Y for s in samples]))) \
                .mapValues(lambda x: [x]) \
                .reduceByKey(lambda a, b: a + b) \
                .map(lambda (region, samples): assignSamples(region, samples)) \
                .map(regionCalPhi) \
                .map(regionCalZ) \
                .map(regionCalBeta) \
                .filter(lambda region: region.beta > option.convTruncation) \
                .cache()

            subRegionsRDD = regionsRDD \
                .filter(lambda region: region.beta < option.convHigh) \
                .flatMap(regionGetSubRegions) \
                .map(regionCalBeta) \
                .filter(checkParamRange) \
                .cache()

            validRegionsRDD = regionsRDD \
                .filter(lambda region: region.beta >= option.convHigh) \
                .map(lambda region: region.hyperParamRange)

            resultsRDD = resultsRDD.union(validRegionsRDD) if resultsRDD else validRegionsRDD

            regionsRDD = subRegionsRDD

            # regionsRDD = subRegionsRDD \
            #     .filter(lambda region: region.beta > option.convTruncation) \
            #     .filter(lambda region: region.beta < option.convHigh)

            # validRegionsRDD = subRegionsRDD \
            #     .filter(lambda region: region.beta >= option.convHigh) \
            #     .map(lambda region: region.hyperParamRange)

            # resultsRDD = resultsRDD.union(validRegionsRDD) if resultsRDD else validRegionsRDD

        results = resultsRDD.collect()
        results.sort(key=lambda hyperParamRange: hyperParamRange.to_array())
        return results
