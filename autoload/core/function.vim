vim9script

import './tensor.vim'
import './context.vim'


export interface HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
endinterface


var last_node_id: number = v:numbermax / 2 - 1

export abstract class Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  def Init(name: string)
    this.name = name
    this.id = last_node_id + 1
    last_node_id = this.id
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return xs
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    return gys
  enddef
endclass
