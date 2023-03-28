import tensorflow_constrained_optimization as tfco
from tensorflow_constrained_optimization import ConstrainedMinimizationProblem
import tensorflow as tf
from VFeatureExtractor import VFeatureExtractor

class Dam2SA(ConstrainedMinimizationProblem):
    def __init__(self, numClasses, subspaceDim, vImages, vLabels, dImages, c=1, mu=1e5, l=1e-2):
        self.c = c
        self.mu = mu
        self.l = l
        self.vFeatureExtractor = VFeatureExtractor(224)
        self.vNumSamples = vImages.shape[0]
        self.dNumSamples = dImages.shape[0]
        self.vImages = vImages
        self.vFeatures = self.vFeatureExtractor(vImages)
        self.vLabels = vLabels
        self.dImages = dImages
        self.vDim = self.vFeatureExtractor.featureSize 
        self.subspaceDim = subspaceDim

        # self.dFeatureExtractor = 
        # self.dFeatures = self.dFeatureExtractor(dImages)?
        self.dFeatures = self.dImages
        self.dDim = 3
        self.labelMask = tf.one_hot(self.vLabels, on_value=1, off_value=-1, depth=numClasses)

        # define trainable parameters
        self.vProj = tf.Variable(tf.random.uniform((self.vDim, self.subspaceDim), -1, 1), trainable=True, name='vProj')
        self.dProj = tf.Variable(tf.random.uniform((self.dDim, self.subspaceDim), -1, 1), trainable=True, name='dProj')
        
        self.vSVMWeight = tf.Variable(tf.random.uniform((subspaceDim, numClasses), -1, 1), trainable=True, name='vSVMWeight')
        self.vSVMBias = tf.Variable(tf.random.uniform((subspaceDim, numClasses), -1, 1), trainable=True, name='vSVMBias')
        self.vErrorLim = tf.Variable(tf.fill([self.vNumSamples, numClasses], 0.1), trainable=True, name='vErrorLim')

        self.dSVMWeight = tf.Variable(tf.random.uniform((subspaceDim, numClasses), -1, 1), trainable=True, name='dSVMWeight')
        self.dSVMBias = tf.Variable(tf.random.uniform((subspaceDim, numClasses), -1, 1), trainable=True, name='dSVMBias')
        self.dErrorLim = tf.Variable(tf.fill([self.dNumSamples, numClasses], 0.1), trainable=True, name='dErrorLim')

    def constraints(self):
        vSub = tf.linalg.matmul(self.vFeatures, self.vProj)
        dSub = tf.linalg.matmul(self.dFeatures, self.dProj)
        vSVMConstraint = 1 - self.vErrorLim - self.labelMask * (tf.linalg.matmul(vSub, self.vSVMWeight) + self.vSVMBias)
        dSVMConstraint = 1 - self.dErrorLim - self.labelMask * (tf.linalg.matmul(dSub, self.dSVMWeight) + self.dSVMBias)
        twoProjConstraintLEQ = tf.linalg.matmul(tf.transpose(vSub), vSub) + tf.linalg.matmul(tf.transpose(dSub), dSub) - tf.eye(self.subspaceDim)
        constraints = tf.stack([
            vSVMConstraint,
            dSVMConstraint,
            -self.vErrorLim,
            -self.dErrorLim,
            twoProjConstraintLEQ,
            -twoProjConstraintLEQ
        ])
    
    def num_constraints(self):
        return 6

    def objective(self):
        vSub = tf.linalg.matmul(self.vFeatures, self.vProj)
        dSub = tf.linalg.matmul(self.dFeatures, self.dProj)
        SVMRegularizerLoss = 0.5 * (tf.reduce_sum(tf.square(self.dSVMWeight)) + tf.reduce_sum(tf.square(self.vSVMWeight)))
        ErrorLimLoss = tf.reduce_sum(self.vErrorLim + self.dErrorLim)
        MMDLoss = tf.norm(tf.reduce_mean(vSub, 0) - tf.reduce_mean(dSub, 0))
        MFCLoss = - tf.linalg.trace(tf.linalg.matmul(tf.transpose(dSub), vSub))
        loss = SVMRegularizerLoss + self.c * ErrorLimLoss + self.mu * MMDLoss + self.l * MFCLoss
        return loss


if __name__ == '__main__':
    bs = 400
    vImages = tf.random.normal((bs, 224, 224, 3))
    dImages = tf.random.normal((bs, 3))
    vLabels = tf.random.uniform((bs, ), 0, 3, tf.int32)
    model = Dam2SA(4, 10, vImages, vLabels, dImages)