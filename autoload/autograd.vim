vim9script


import './core/tensor.vim'
export const Tensor = tensor.Tensor
export const Clone = tensor.Clone
export const Detach = tensor.Detach
export const EmptyTensor = tensor.EmptyTensor
export const IsTensor = tensor.IsTensor
export const AsTensor = tensor.AsTensor
export const Zeros = tensor.Zeros
export const ZerosLike = tensor.ZerosLike
export const Ones = tensor.Ones
export const OnesLike = tensor.OnesLike

import './core/backward.vim'
export const Backward = backward.Backward

import './core/function.vim'
export const Function = function.Function

import './core/callfunc.vim'
export const CallFunction = callfunc.CallFunction

import './core/grad.vim'
export const Grad = grad.Grad

import './core/random.vim'
export const ManualSeed = random.ManualSeed
export const Random = random.Random
export const Rand = random.Rand
export const Uniform = random.Uniform
export const Randn = random.Randn
export const Normal = random.Normal

import './core/engine.vim'
export const Elementwise = engine.Elementwise

import './core/context.vim'
export const NoGrad = context.NoGrad

import './functions/abs.vim'
export const Abs = abs.Abs

import './functions/add.vim'
export const Add = add.Add

import './functions/broadcast_to.vim'
export const BroadcastTo = broadcast_to.BroadcastTo

import './functions/cos.vim'
export const Cos = cos.Cos

import './functions/div.vim'
export const Div = div.Div

import './functions/exp.vim'
export const Exp = exp.Exp

import './functions/log.vim'
export const Log = log.Log

import './functions/matmul.vim'
export const Matmul = matmul.Matmul

import './functions/max.vim'
export const Max = max.Max

import './functions/maximum.vim'
export const Maximum = maximum.Maximum

import './functions/mul.vim'
export const Mul = mul.Mul

import './functions/pow.vim'
export const Pow = pow.Pow
export const Sqrt = pow.Sqrt

import './functions/reshape.vim'
export const Reshape = reshape.Reshape
export const Flatten = reshape.Flatten

import './functions/sign.vim'
export const Sign = sign.Sign

import './functions/sin.vim'
export const Sin = sin.Sin

import './functions/sub.vim'
export const Sub = sub.Sub

import './functions/sum.vim'
export const Sum = sum.Sum

import './functions/sum_to.vim'
export const SumTo = sum_to.SumTo

import './functions/tanh.vim'
export const Tanh = tanh.Tanh

import './functions/transpose.vim'
export const Transpose = transpose.Transpose

import './utils/graph.vim'
export const DumpGraph = graph.DumpGraph

import './utils/data.vim'
export const Shuffle = data.Shuffle

import './utils/check.vim'
export const NumericalGrad = check.NumericalGrad
export const GradCheck = check.GradCheck

import './utils/math.vim'
export const Pi = math.Pi

import './utils/matrix.vim'
export const ShapeToSize = matrix.ShapeToSize
export const AsList = matrix.AsList
export const CreateVector = matrix.CreateVector
export const GetMatrixShape = matrix.GetMatrixShape
export const SqueezeRightShape = matrix.SqueezeRightShape
export const SqueezeLeftShape = matrix.SqueezeLeftShape

import './utils/system.vim'
export const Error = system.Error

defcompile
