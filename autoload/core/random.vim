vim9script

import '../utils/matrix.vim'
import '../utils/math.vim'
import './tensor.vim'


var random_seed = srand()

export def ManualSeed(seed: number)
  random_seed = srand(seed)
enddef


export def Random(): float
  # it returns random value from 0.0 to 1.0.
  return rand(random_seed) / 4294967295.0
enddef


def BoxMuller(u1: float, u2: float): float
  return sqrt(-2 * log(u1)) * cos(2 * math.Pi() * u2)
enddef


export def Rand(...shape: list<number>): tensor.Tensor
  var _shape = len(shape) > 0 ? shape : [1]
  var size = matrix.ShapeToSize(_shape)
  var data = map(repeat([0.0], size), (..._): float => {
      return Random()
    })
  return tensor.Tensor.new(data, _shape)
enddef


export def Uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: list<number> = [1]): tensor.Tensor
  var size = matrix.ShapeToSize(shape)
  var data = map(
    repeat([0.0], size), (..._): float => {
        return low + (high - low) * Random()
      })
  return tensor.Tensor.new(data, shape)
enddef


export def Randn(...shape: list<number>): tensor.Tensor
  var _shape = len(shape) > 0 ? shape : [1]
  var size = matrix.ShapeToSize(_shape)
  var data = map(
    repeat([0.0], size), (..._): float => {
        return BoxMuller(Random(), Random())
      })
  return tensor.Tensor.new(data, _shape)
enddef


export def Normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: list<number> = [1]): tensor.Tensor
  var size = matrix.ShapeToSize(shape)
  var data = map(
    repeat([0.0], size), (..._): float => {
        return std * BoxMuller(Random(), Random()) + mean
      })
  return tensor.Tensor.new(data, shape)
enddef
