vim9script

import '../utils/system.vim'
import '../utils/matrix.vim'


var last_node_id: number = 0


export class Tensor
  public this.name: string
  public this.gen: number
  public this.id: number

  public this.data: list<float>
  public this.grad: any
  public this.parent_fn: any

  this.shape: list<number>

  def new(raw_data: any, shape: list<number> = null_list)
    if shape == null_list
      var _data = matrix.AsList(deepcopy(raw_data))
      this.shape = matrix.GetMatrixShape(_data)

      this.data = map(
        flattennew(_data), (_, v): float => v * 1.0)
    else
      this.data = raw_data
      this.shape = shape
    endif

    if len(this.data) != matrix.ShapeToSize(this.shape)
      system.Error('Invalid matrix shape.')
    endif

    this.id = last_node_id + 1
    last_node_id = this.id

    this.grad = null
    this.parent_fn = null
  enddef

  def Numel(): number
    return len(this.data)
  enddef

  def Dim(): number
    return len(this.shape)
  enddef

  def Empty(): bool
    return this.Numel() == 0
  enddef

  def ClearGrad()
    this.grad = null
  enddef

  def SetName(name: string)
    this.name = name
  enddef
endclass


export def Clone(x: Tensor): Tensor
  return Tensor.new(copy(x.data), copy(x.shape))
enddef


export def Detach(x: Tensor): Tensor
  # It returns a new tensor detached from the current graph.
  # However, returned tensor shares the same data and shape attribute.
  return Tensor.new(x.data, x.shape)
enddef


export const EmptyTensor = Tensor.new([])


const type_of_tensor = type(EmptyTensor)
export def IsTensor(data: any): bool
  return type(data) == type_of_tensor
enddef


export def AsTensor(data: any): Tensor
  return IsTensor(data) ? data : Tensor.new(data)
enddef


export def Zeros(shape: list<number>): Tensor
  var size = matrix.ShapeToSize(shape)
  if size == 0
    system.Error('axis without element is invalid.')
  endif
  return Tensor.new(matrix.CreateVector(size, 0.0), shape)
enddef


export def ZerosLike(tensor: Tensor): Tensor
  return Tensor.new(
    matrix.CreateVector(len(tensor.data), 0.0),
    tensor.shape)
enddef


export def Ones(shape: list<number>): Tensor
  var size = matrix.ShapeToSize(shape)
  if size == 0
    system.Error('axis without element is invalid.')
  endif
  return Tensor.new(matrix.CreateVector(size, 1.0), shape)
enddef


export def OnesLike(tensor: Tensor): Tensor
  return Tensor.new(
    matrix.CreateVector(len(tensor.data), 1.0),
    tensor.shape)
enddef
